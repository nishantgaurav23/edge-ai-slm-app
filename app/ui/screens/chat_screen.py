import logging
import threading
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.label import MDLabel
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDRaisedButton, MDIconButton
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.metrics import dp

from app.core.inference_engine import InferenceEngine
from app.core.context_manager import ContextManager
from app.core.data_store import DataStore
from app.services.model_loader import ModelLoader
from app.services.hardware_monitor import HardwareMonitor, BatteryOptimizer

logger = logging.getLogger(__name__)

# Load KV at module level (only once)
# Define the KV string for the chat UI
KV = '''
<ChatScreen>:
    MDBoxLayout:
        orientation: "vertical"

        # Top bar with back button and title
        MDTopAppBar:
            id: top_bar
            title: "AI Chat"
            left_action_items: [["arrow-left", lambda x: root.go_back()]]
            right_action_items: [["reload", lambda x: root.reload_model()]]
            md_bg_color: app.theme_cls.primary_color

        # Messages scroll area
        MDScrollView:
            id: scroll_view
            do_scroll_x: False

            MDBoxLayout:
                id: messages_box
                orientation: "vertical"
                spacing: dp(10)
                padding: dp(10)
                adaptive_height: True

        # Input area
        MDBoxLayout:
            orientation: "horizontal"
            size_hint_y: None
            height: dp(60)
            padding: dp(10)
            spacing: dp(10)

            MDTextField:
                id: input_field
                hint_text: "Type your message..."
                multiline: False
                size_hint_x: 0.8
                on_text_validate: root.send_message()

            MDIconButton:
                icon: "send"
                size_hint_x: 0.2
                on_release: root.send_message()
                md_bg_color: app.theme_cls.primary_color
'''

# Load KV at module level
Builder.load_string(KV)


class ChatScreen(MDScreen):
    """
    Chat screen with integration to Edge AI SLM backend.

    Features:
    - Displays chat messages
    - Sends user input to InferenceEngine
    - Uses ContextManager for conversation history
    - Battery-aware processing
    - Lazy model loading
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize backend services
        self.inference_engine = InferenceEngine.get_instance()
        self.model_loader = ModelLoader()
        self.data_store = DataStore()
        self.conversation_id = self.data_store.create_conversation("Edge AI Chat")
        self.context_manager = ContextManager(
            max_tokens=2048,
            system_prompt="",  # Empty system prompt for TinyLlama
            data_store=self.data_store,
            conversation_id=self.conversation_id
        )

        # Model configuration
        self.model_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        self.model_loaded = False

        # UI state
        self.waiting_for_response = False

    def on_enter(self):
        """Called when screen is displayed."""
        # Delay until widgets are built
        Clock.schedule_once(lambda dt: self._on_enter_delayed(), 0.1)

    def _on_enter_delayed(self):
        """Called after widgets are ready."""
        # Check battery and hardware
        self._display_system_info()

        # Try to lazy load model if not already loaded
        if not self.model_loaded:
            self._lazy_load_model()

    def go_back(self):
        """Navigate back to home screen."""
        from kivy.app import App
        app = App.get_running_app()
        if app and app.root:
            app.root.current = 'home'

    def _display_system_info(self):
        """Display system and battery info."""
        ram_gb = HardwareMonitor.get_total_ram_gb()
        is_low_end = HardwareMonitor.is_low_end_device()
        battery_percent = HardwareMonitor.get_battery_percent()
        is_charging = HardwareMonitor.is_charging()
        power_mode = BatteryOptimizer.get_power_mode()

        info_text = (
            f"Device: {ram_gb:.1f}GB RAM {'(Low-end)' if is_low_end else '(High-end)'}\n"
            f"Battery: {battery_percent:.0f}% {'(Charging)' if is_charging else ''}\n"
            f"Power mode: {power_mode}"
        )

        self._add_system_message(info_text)
        logger.info(info_text)

    def _lazy_load_model(self):
        """Lazy load the model on-demand."""
        self._add_system_message("Loading AI model... This may take a moment.")

        # Run in background thread to avoid blocking UI
        def load_thread():
            try:
                # Check if model exists
                model_path = self.model_loader.get_model_path(self.model_filename)

                if not model_path:
                    error_msg = (
                        f"Model '{self.model_filename}' not found.\n"
                        f"Please download a GGUF model and place it in the 'models/' directory.\n"
                        f"Suggested model: TinyLlama-1.1B-Chat-v1.0-GGUF (Q4_K_M)"
                    )
                    Clock.schedule_once(lambda dt: self._add_system_message(error_msg), 0)
                    return

                # Detect hardware and set quantization
                n_gpu_layers = 0  # CPU only for now
                n_ctx = 2048

                # Adjust context based on hardware
                if HardwareMonitor.is_low_end_device():
                    n_ctx = 1024
                    logger.info("Low-end device detected, reducing context to 1024")

                # Load model
                self.inference_engine.load_model(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers
                )

                self.model_loaded = True
                Clock.schedule_once(
                    lambda dt: self._add_system_message("Model loaded successfully! You can start chatting."),
                    0
                )

            except Exception as e:
                error_msg = f"Failed to load model: {str(e)}"
                logger.error(error_msg)
                Clock.schedule_once(lambda dt: self._add_system_message(error_msg), 0)

        threading.Thread(target=load_thread, daemon=True).start()

    def reload_model(self):
        """Reload the model (useful for testing or after memory pressure)."""
        self._add_system_message("Reloading model...")
        self.inference_engine.unload_model()
        self.model_loaded = False
        self._lazy_load_model()

    def send_message(self):
        """Send user message and get AI response."""
        logger.info("send_message called")

        if self.waiting_for_response:
            logger.info("Already waiting for response, ignoring")
            return

        # Check if widgets are ready
        if 'input_field' not in self.ids:
            logger.warning("Input field not ready yet")
            return

        # Get input field widget
        input_field = self.ids.input_field
        user_message = input_field.text.strip()

        logger.info(f"User message: '{user_message}'")

        if not user_message:
            logger.info("Empty message, ignoring")
            return

        # Clear input
        input_field.text = ""

        # Display user message
        logger.info("Adding user message to UI")
        self._add_user_message(user_message)

        # Add to context
        self.context_manager.add_message("user", user_message)

        # Check if model is loaded
        if not self.model_loaded:
            self._add_system_message("Model not loaded. Please wait for it to load.")
            return

        # Check battery mode
        if BatteryOptimizer.should_throttle():
            self._add_system_message("Low battery detected. Response may be slower.")

        # Generate response in background
        self.waiting_for_response = True
        self._add_ai_message("Thinking...")

        def generate_thread():
            try:
                # Get regular context (keep it simple for TinyLlama)
                context = self.context_manager.get_context()

                # Build prompt
                prompt = self._build_prompt(context)
                logger.info(f"Full prompt length: {len(prompt)} chars")

                # Generate response with conservative parameters for better accuracy
                response = self.inference_engine.generate_response(
                    prompt=prompt,
                    max_tokens=100,  # Keep responses short
                    stream=False,
                    temperature=0.3,  # Low temperature for more focused responses
                    top_p=0.9,        # Nucleus sampling
                    repeat_penalty=1.3,  # Strong penalty against repetition
                    stop=["</s>", "<|user|>", "<|system|>", "\n\n", "User:", "Assistant:"]
                )

                # Extract text from response
                if isinstance(response, dict):
                    ai_text = response.get('choices', [{}])[0].get('text', '').strip()
                else:
                    ai_text = str(response).strip()

                # Clean up response - remove special tokens and extra whitespace
                ai_text = ai_text.replace('</s>', '').replace('<|user|>', '').replace('<|system|>', '').replace('<|assistant|>', '')
                ai_text = ai_text.strip()

                # If response is empty, provide fallback
                if not ai_text:
                    ai_text = "I apologize, I couldn't generate a proper response. Please try rephrasing your question."

                # Add to context
                self.context_manager.add_message("assistant", ai_text)

                # Update UI
                Clock.schedule_once(lambda dt: self._update_last_ai_message(ai_text), 0)

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg)
                Clock.schedule_once(lambda dt: self._update_last_ai_message(error_msg), 0)

            finally:
                self.waiting_for_response = False

        threading.Thread(target=generate_thread, daemon=True).start()

    def _build_prompt(self, context):
        """Build prompt from context messages using TinyLlama's chat format with few-shot examples."""
        # Add instruction and few-shot examples to guide TinyLlama
        prompt = """<|system|>
You are a helpful AI assistant. Answer questions directly and stay on topic.</s>
<|user|>
Hello</s>
<|assistant|>
Hello! How can I help you today?</s>
<|user|>
What is 2+2?</s>
<|assistant|>
2+2 equals 4.</s>
"""

        for msg in context:
            role = msg['role']
            content = msg['content']

            # Skip system messages (we added our own above)
            if role == 'system':
                continue
            elif role == 'user':
                prompt += f"<|user|>\n{content}</s>\n"
            elif role == 'assistant':
                prompt += f"<|assistant|>\n{content}</s>\n"

        # Add the assistant prompt for generation
        prompt += "<|assistant|>\n"

        # Log the prompt for debugging
        logger.info(f"Generated prompt:\n{prompt[:200]}...")

        return prompt

    def _add_user_message(self, text):
        """Add a user message bubble to the chat."""
        logger.info(f"_add_user_message called with: '{text}'")
        if 'messages_box' not in self.ids:
            logger.warning("messages_box not in ids yet")
            return
        messages_box = self.ids.messages_box
        logger.info("Adding message to messages_box")

        msg_label = MDLabel(
            text=text,
            size_hint_y=None,
            height=dp(40),
            padding=(dp(10), dp(5)),
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),
            halign="right"
        )
        msg_label.bind(
            texture_size=lambda instance, value: setattr(instance, 'height', value[1] + dp(10))
        )

        messages_box.add_widget(msg_label)
        self._scroll_to_bottom()

    def _add_ai_message(self, text):
        """Add an AI message bubble to the chat."""
        if 'messages_box' not in self.ids:
            return
        messages_box = self.ids.messages_box

        msg_label = MDLabel(
            text=text,
            size_hint_y=None,
            height=dp(40),
            padding=(dp(10), dp(5)),
            theme_text_color="Custom",
            text_color=(0.3, 0.7, 1, 1),
            halign="left"
        )
        msg_label.bind(
            texture_size=lambda instance, value: setattr(instance, 'height', value[1] + dp(10))
        )

        messages_box.add_widget(msg_label)
        self.last_ai_label = msg_label
        self._scroll_to_bottom()

    def _add_system_message(self, text):
        """Add a system message to the chat."""
        if 'messages_box' not in self.ids:
            return
        messages_box = self.ids.messages_box

        msg_label = MDLabel(
            text=f"[System] {text}",
            size_hint_y=None,
            height=dp(40),
            padding=(dp(10), dp(5)),
            theme_text_color="Custom",
            text_color=(0.7, 0.7, 0.7, 1),
            halign="center",
            italic=True
        )
        msg_label.bind(
            texture_size=lambda instance, value: setattr(instance, 'height', value[1] + dp(10))
        )

        messages_box.add_widget(msg_label)
        self._scroll_to_bottom()

    def _update_last_ai_message(self, text):
        """Update the last AI message (replace 'Thinking...')."""
        if hasattr(self, 'last_ai_label'):
            self.last_ai_label.text = text

    def _scroll_to_bottom(self):
        """Scroll chat to bottom."""
        if 'scroll_view' not in self.ids:
            return

        def do_scroll(dt):
            if 'scroll_view' in self.ids:
                scroll_view = self.ids.scroll_view
                scroll_view.scroll_y = 0

        Clock.schedule_once(do_scroll, 0.1)
