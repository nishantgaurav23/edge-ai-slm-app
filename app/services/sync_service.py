"""
Sync Service - Offline-First Data Synchronization for Edge AI SLM App

Provides:
- Local-first storage with optional cloud sync
- Conflict resolution (local changes win by default)
- User-controlled sync with permission
"""

import logging
import json
import hashlib
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Sync status for data items."""
    LOCAL_ONLY = "local_only"
    SYNCED = "synced"
    PENDING_UPLOAD = "pending_upload"
    PENDING_DOWNLOAD = "pending_download"
    CONFLICT = "conflict"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    MERGE = "merge"
    MANUAL = "manual"


class SyncMetadata:
    """Metadata for sync tracking."""
    
    def __init__(
        self,
        item_id: str,
        local_hash: str,
        remote_hash: Optional[str] = None,
        last_synced: Optional[float] = None,
        last_modified: Optional[float] = None,
        status: SyncStatus = SyncStatus.LOCAL_ONLY
    ):
        self.item_id = item_id
        self.local_hash = local_hash
        self.remote_hash = remote_hash
        self.last_synced = last_synced
        self.last_modified = last_modified or time.time()
        self.status = status
    
    def to_dict(self) -> Dict:
        return {
            'item_id': self.item_id,
            'local_hash': self.local_hash,
            'remote_hash': self.remote_hash,
            'last_synced': self.last_synced,
            'last_modified': self.last_modified,
            'status': self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SyncMetadata':
        return cls(
            item_id=data['item_id'],
            local_hash=data['local_hash'],
            remote_hash=data.get('remote_hash'),
            last_synced=data.get('last_synced'),
            last_modified=data.get('last_modified'),
            status=SyncStatus(data.get('status', 'local_only'))
        )


class SyncEngine:
    """
    Handles offline-first synchronization with optional cloud backup.
    
    Principles:
    - Local data is always the source of truth
    - Sync only happens with user permission
    - Conflicts default to local-wins resolution
    """
    
    def __init__(
        self,
        data_store,
        cloud_adapter=None,
        conflict_resolution: ConflictResolution = ConflictResolution.LOCAL_WINS
    ):
        self.data_store = data_store
        self.cloud_adapter = cloud_adapter  # Plugin for cloud storage (Firebase, S3, etc.)
        self.conflict_resolution = conflict_resolution
        self._sync_metadata: Dict[str, SyncMetadata] = {}
        self._sync_enabled = False
        self._user_consented = False
    
    def enable_sync(self, user_consent: bool = False):
        """
        Enable synchronization with user consent.
        
        Args:
            user_consent: User explicitly agreed to sync data to cloud
        """
        if not user_consent:
            logger.warning("Sync requires explicit user consent")
            return False
        
        if not self.cloud_adapter:
            logger.warning("No cloud adapter configured")
            return False
        
        self._user_consented = True
        self._sync_enabled = True
        logger.info("Sync enabled with user consent")
        return True
    
    def disable_sync(self):
        """Disable synchronization."""
        self._sync_enabled = False
        logger.info("Sync disabled")
    
    @staticmethod
    def compute_hash(data: Any) -> str:
        """Compute hash of data for change detection."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    def mark_for_sync(self, item_id: str, data: Any):
        """Mark an item as pending sync."""
        local_hash = self.compute_hash(data)
        
        if item_id in self._sync_metadata:
            metadata = self._sync_metadata[item_id]
            metadata.local_hash = local_hash
            metadata.last_modified = time.time()
            metadata.status = SyncStatus.PENDING_UPLOAD
        else:
            metadata = SyncMetadata(
                item_id=item_id,
                local_hash=local_hash,
                status=SyncStatus.PENDING_UPLOAD
            )
            self._sync_metadata[item_id] = metadata
    
    def get_pending_items(self) -> List[str]:
        """Get list of items pending sync."""
        return [
            item_id for item_id, meta in self._sync_metadata.items()
            if meta.status == SyncStatus.PENDING_UPLOAD
        ]
    
    def sync_item(self, item_id: str, data: Any) -> bool:
        """
        Sync a single item to cloud.
        
        Returns:
            True if sync successful
        """
        if not self._sync_enabled:
            logger.debug("Sync disabled, skipping")
            return False
        
        if not self.cloud_adapter:
            return False
        
        metadata = self._sync_metadata.get(item_id)
        if not metadata:
            self.mark_for_sync(item_id, data)
            metadata = self._sync_metadata[item_id]
        
        try:
            # Check for conflicts first
            remote_data = self.cloud_adapter.get(item_id)
            
            if remote_data:
                remote_hash = self.compute_hash(remote_data)
                
                # Conflict: both local and remote changed
                if (metadata.remote_hash and 
                    remote_hash != metadata.remote_hash and 
                    metadata.local_hash != metadata.remote_hash):
                    
                    resolved_data = self._resolve_conflict(data, remote_data)
                    data = resolved_data
            
            # Upload to cloud
            success = self.cloud_adapter.put(item_id, data)
            
            if success:
                new_hash = self.compute_hash(data)
                metadata.remote_hash = new_hash
                metadata.local_hash = new_hash
                metadata.last_synced = time.time()
                metadata.status = SyncStatus.SYNCED
                logger.info(f"Synced item: {item_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Sync failed for {item_id}: {e}")
            return False
    
    def _resolve_conflict(self, local_data: Any, remote_data: Any) -> Any:
        """
        Resolve conflict between local and remote data.
        Default: local wins.
        """
        if self.conflict_resolution == ConflictResolution.LOCAL_WINS:
            logger.info("Conflict resolved: local wins")
            return local_data
        
        elif self.conflict_resolution == ConflictResolution.REMOTE_WINS:
            logger.info("Conflict resolved: remote wins")
            return remote_data
        
        elif self.conflict_resolution == ConflictResolution.MERGE:
            # Simple merge strategy for dict-like data
            if isinstance(local_data, dict) and isinstance(remote_data, dict):
                merged = {**remote_data, **local_data}
                logger.info("Conflict resolved: merged")
                return merged
            return local_data
        
        else:
            # Manual resolution - keep local for now, flag for user review
            logger.warning("Conflict requires manual resolution, keeping local")
            return local_data
    
    def sync_all(self) -> Dict[str, bool]:
        """
        Sync all pending items.
        
        Returns:
            Dict mapping item_id to success status
        """
        if not self._sync_enabled:
            return {}
        
        results = {}
        pending = self.get_pending_items()
        
        for item_id in pending:
            # Get data from data store
            # This is a simplified example - real implementation would
            # fetch the actual data from the data store
            results[item_id] = False  # Placeholder
        
        logger.info(f"Sync complete: {sum(results.values())}/{len(results)} succeeded")
        return results
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get overall sync status."""
        pending = len(self.get_pending_items())
        synced = len([m for m in self._sync_metadata.values() if m.status == SyncStatus.SYNCED])
        
        return {
            'enabled': self._sync_enabled,
            'user_consented': self._user_consented,
            'pending_count': pending,
            'synced_count': synced,
            'has_cloud_adapter': self.cloud_adapter is not None
        }


class CloudAdapter:
    """
    Base class for cloud storage adapters.
    Implement this for Firebase, S3, or other backends.
    """
    
    def get(self, item_id: str) -> Optional[Any]:
        """Get item from cloud storage."""
        raise NotImplementedError
    
    def put(self, item_id: str, data: Any) -> bool:
        """Put item to cloud storage."""
        raise NotImplementedError
    
    def delete(self, item_id: str) -> bool:
        """Delete item from cloud storage."""
        raise NotImplementedError
    
    def list(self) -> List[str]:
        """List all items in cloud storage."""
        raise NotImplementedError
