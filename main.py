import logging
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from kivymd.uix.screen import MDScreen

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.ui.screens.chat_screen import ChatScreen

class HomeScreen(MDScreen):
    pass

class EdgeAIApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Teal"
        
        # Define Screen Manager and Screens
        sm = ScreenManager()
        
        # Home Screen
        home_screen = HomeScreen(name='home')
        home_screen.add_widget(Builder.load_string('''
MDBoxLayout:
    orientation: "vertical"
    padding: dp(20)
    spacing: dp(20)

    MDLabel:
        text: "Edge AI SLM App"
        halign: "center"
        font_style: "H3"
        theme_text_color: "Primary"

    MDRaisedButton:
        text: "Start Chat"
        pos_hint: {"center_x": .5}
        on_release: app.root.current = 'chat'
'''))
        sm.add_widget(home_screen)

        # Chat Screen
        sm.add_widget(ChatScreen(name='chat'))
        
        return sm

if __name__ == '__main__':
    EdgeAIApp().run()
