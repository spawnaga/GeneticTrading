
#!/usr/bin/env python
"""
Model Manager Utility
=====================

Professional model management with versioning, backup, and restoration capabilities.
"""

import os
import json
import shutil
import datetime
from pathlib import Path
from typing import Dict, List, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Professional model management system."""
    
    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {
            'ga': self.base_dir / 'ga_models',
            'ppo': self.base_dir / 'ppo_models',
            'checkpoints': self.base_dir / 'checkpoints',
            'backups': self.base_dir / 'backups',
            'exports': self.base_dir / 'exports'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)
        
        self.metadata_file = self.base_dir / 'model_metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load model metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'models': {}, 'backups': {}, 'version': '1.0'}
    
    def _save_metadata(self):
        """Save model metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def save_model(self, model, model_name: str, model_type: str, 
                   metrics: Optional[Dict] = None, description: str = ""):
        """
        Save model with metadata and backup management.
        
        Args:
            model: PyTorch model to save
            model_name: Name of the model
            model_type: Type ('ga' or 'ppo')
            metrics: Performance metrics dict
            description: Model description
        """
        if model_type not in self.subdirs:
            raise ValueError(f"Invalid model type: {model_type}")
        
        model_dir = self.subdirs[model_type]
        model_path = model_dir / f"{model_name}.pth"
        
        # Create backup if model exists
        if model_path.exists():
            self._create_backup(model_path, model_name, model_type)
        
        # Save model
        if hasattr(model, 'save_model'):
            model.save_model(str(model_path))
        else:
            torch.save(model.state_dict(), model_path)
        
        # Update metadata
        timestamp = datetime.datetime.now()
        self.metadata['models'][model_name] = {
            'type': model_type,
            'path': str(model_path),
            'created': timestamp.isoformat(),
            'description': description,
            'metrics': metrics or {},
            'version': len(self.metadata.get('models', {})) + 1
        }
        
        self._save_metadata()
        logger.info(f"Saved {model_type} model '{model_name}' to {model_path}")
    
    def _create_backup(self, model_path: Path, model_name: str, model_type: str):
        """Create backup of existing model."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{model_name}_{timestamp}.pth"
        backup_path = self.subdirs['backups'] / backup_name
        
        shutil.copy2(model_path, backup_path)
        
        # Update backup metadata
        if 'backups' not in self.metadata:
            self.metadata['backups'] = {}
        
        self.metadata['backups'][backup_name] = {
            'original_name': model_name,
            'type': model_type,
            'backup_time': timestamp,
            'path': str(backup_path)
        }
        
        logger.info(f"Created backup: {backup_name}")
    
    def load_model(self, model, model_name: str) -> bool:
        """
        Load model by name.
        
        Args:
            model: Model instance to load state into
            model_name: Name of the model to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        if model_name not in self.metadata['models']:
            logger.warning(f"Model '{model_name}' not found in metadata")
            return False
        
        model_info = self.metadata['models'][model_name]
        model_path = Path(model_info['path'])
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            if hasattr(model, 'load_model'):
                model.load_model(str(model_path))
            else:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            logger.info(f"Loaded model '{model_name}' from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            return False
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict]:
        """List all models, optionally filtered by type."""
        models = []
        for name, info in self.metadata['models'].items():
            if model_type is None or info['type'] == model_type:
                models.append({
                    'name': name,
                    'type': info['type'],
                    'created': info['created'],
                    'description': info['description'],
                    'metrics': info.get('metrics', {}),
                    'version': info.get('version', 1)
                })
        
        return sorted(models, key=lambda x: x['created'], reverse=True)
    
    def list_backups(self, model_name: Optional[str] = None) -> List[Dict]:
        """List backups, optionally filtered by model name."""
        backups = []
        for backup_name, info in self.metadata.get('backups', {}).items():
            if model_name is None or info['original_name'] == model_name:
                backups.append({
                    'backup_name': backup_name,
                    'original_name': info['original_name'],
                    'type': info['type'],
                    'backup_time': info['backup_time'],
                    'path': info['path']
                })
        
        return sorted(backups, key=lambda x: x['backup_time'], reverse=True)
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore a model from backup."""
        if backup_name not in self.metadata.get('backups', {}):
            logger.error(f"Backup '{backup_name}' not found")
            return False
        
        backup_info = self.metadata['backups'][backup_name]
        backup_path = Path(backup_info['path'])
        
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        # Get original model info
        original_name = backup_info['original_name']
        model_type = backup_info['type']
        model_dir = self.subdirs[model_type]
        model_path = model_dir / f"{original_name}.pth"
        
        # Create backup of current model if it exists
        if model_path.exists():
            self._create_backup(model_path, original_name, model_type)
        
        # Restore from backup
        shutil.copy2(backup_path, model_path)
        logger.info(f"Restored '{original_name}' from backup '{backup_name}'")
        
        return True
    
    def cleanup_old_backups(self, keep_last: int = 10):
        """Clean up old backups, keeping only the most recent."""
        backups = self.list_backups()
        
        # Group by original model name
        backup_groups = {}
        for backup in backups:
            name = backup['original_name']
            if name not in backup_groups:
                backup_groups[name] = []
            backup_groups[name].append(backup)
        
        # Remove old backups
        for model_name, model_backups in backup_groups.items():
            model_backups.sort(key=lambda x: x['backup_time'], reverse=True)
            
            if len(model_backups) > keep_last:
                old_backups = model_backups[keep_last:]
                for backup in old_backups:
                    backup_path = Path(backup['path'])
                    if backup_path.exists():
                        backup_path.unlink()
                    
                    # Remove from metadata
                    if backup['backup_name'] in self.metadata['backups']:
                        del self.metadata['backups'][backup['backup_name']]
                    
                    logger.info(f"Removed old backup: {backup['backup_name']}")
        
        self._save_metadata()
    
    def export_model(self, model_name: str, export_format: str = 'pytorch') -> Optional[str]:
        """Export model for deployment."""
        if model_name not in self.metadata['models']:
            logger.error(f"Model '{model_name}' not found")
            return None
        
        model_info = self.metadata['models'][model_name]
        source_path = Path(model_info['path'])
        
        if not source_path.exists():
            logger.error(f"Model file not found: {source_path}")
            return None
        
        # Create export filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{model_name}_export_{timestamp}.pth"
        export_path = self.subdirs['exports'] / export_name
        
        # Copy model
        shutil.copy2(source_path, export_path)
        
        # Create export metadata
        export_metadata = {
            'original_model': model_name,
            'export_time': timestamp,
            'format': export_format,
            'metrics': model_info.get('metrics', {}),
            'description': model_info.get('description', '')
        }
        
        metadata_path = export_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(export_metadata, f, indent=2, default=str)
        
        logger.info(f"Exported model '{model_name}' to {export_path}")
        return str(export_path)


# CLI interface for model management
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Management CLI")
    parser.add_argument("--models-dir", default="./models", help="Models directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List models")
    list_parser.add_argument("--type", choices=["ga", "ppo"], help="Filter by model type")
    
    # Backup command
    backup_parser = subparsers.add_parser("backups", help="List backups")
    backup_parser.add_argument("--model", help="Filter by model name")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup old backups")
    cleanup_parser.add_argument("--keep", type=int, default=10, help="Number of backups to keep")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("model_name", help="Model name to export")
    
    args = parser.parse_args()
    
    manager = ModelManager(args.models_dir)
    
    if args.command == "list":
        models = manager.list_models(args.type)
        print(f"\n{'Name':<20} {'Type':<8} {'Created':<20} {'Description'}")
        print("-" * 70)
        for model in models:
            print(f"{model['name']:<20} {model['type']:<8} {model['created'][:19]:<20} {model['description']}")
    
    elif args.command == "backups":
        backups = manager.list_backups(args.model)
        print(f"\n{'Backup Name':<30} {'Original':<20} {'Type':<8} {'Created'}")
        print("-" * 80)
        for backup in backups:
            print(f"{backup['backup_name']:<30} {backup['original_name']:<20} {backup['type']:<8} {backup['backup_time']}")
    
    elif args.command == "cleanup":
        manager.cleanup_old_backups(args.keep)
        print(f"Cleaned up old backups, keeping last {args.keep}")
    
    elif args.command == "export":
        export_path = manager.export_model(args.model_name)
        if export_path:
            print(f"Model exported to: {export_path}")
    
    else:
        parser.print_help()
