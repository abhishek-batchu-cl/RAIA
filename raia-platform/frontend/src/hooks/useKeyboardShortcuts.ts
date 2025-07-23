import { useEffect, useCallback } from 'react';

export interface KeyboardShortcut {
  key: string;
  description: string;
  action: () => void;
  category?: 'navigation' | 'data' | 'editing' | 'view' | 'help';
  enabled?: boolean;
}

interface UseKeyboardShortcutsOptions {
  shortcuts: KeyboardShortcut[];
  enabled?: boolean;
  preventDefault?: boolean;
}

const useKeyboardShortcuts = ({ 
  shortcuts, 
  enabled = true, 
  preventDefault = true 
}: UseKeyboardShortcutsOptions) => {
  
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (!enabled) return;

    // Don't trigger shortcuts when typing in inputs
    const target = event.target as HTMLElement;
    if (target.tagName === 'INPUT' || 
        target.tagName === 'TEXTAREA' || 
        target.contentEditable === 'true') {
      return;
    }

    const shortcut = shortcuts.find(s => {
      if (s.enabled === false) return false;
      
      const keys = s.key.toLowerCase().split('+').map(k => k.trim());
      const eventKeys: string[] = [];

      // Build event key combination
      if (event.ctrlKey || event.metaKey) eventKeys.push('cmd');
      if (event.shiftKey) eventKeys.push('shift');
      if (event.altKey) eventKeys.push('alt');
      
      // Add the main key
      const mainKey = event.key.toLowerCase();
      if (!['control', 'shift', 'alt', 'meta'].includes(mainKey)) {
        eventKeys.push(mainKey === ' ' ? 'space' : mainKey);
      }

      // Check if keys match
      return keys.length === eventKeys.length && 
             keys.every(key => eventKeys.includes(key));
    });

    if (shortcut) {
      if (preventDefault) {
        event.preventDefault();
        event.stopPropagation();
      }
      shortcut.action();
    }
  }, [shortcuts, enabled, preventDefault]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return shortcuts;
};

// Pre-defined shortcuts for the RAIA platform
export const useRAIAShortcuts = (callbacks: {
  onSearch?: () => void;
  onHelp?: () => void;
  onToggleSidebar?: () => void;
  onDuplicateView?: () => void;
  onCloseModal?: () => void;
  onRefresh?: () => void;
  onExport?: () => void;
  onSave?: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  onFullscreen?: () => void;
  onNotifications?: () => void;
  onSettings?: () => void;
}) => {
  const shortcuts: KeyboardShortcut[] = [
    // Navigation shortcuts
    {
      key: 'cmd+k',
      description: 'Open quick search',
      action: callbacks.onSearch || (() => {}),
      category: 'navigation'
    },
    {
      key: 'cmd+/',
      description: 'Show help and shortcuts',
      action: callbacks.onHelp || (() => {}),
      category: 'help'
    },
    {
      key: 'cmd+b',
      description: 'Toggle sidebar',
      action: callbacks.onToggleSidebar || (() => {}),
      category: 'view'
    },
    {
      key: 'cmd+d',
      description: 'Duplicate current view',
      action: callbacks.onDuplicateView || (() => {}),
      category: 'view'
    },
    {
      key: 'escape',
      description: 'Close modal or dialog',
      action: callbacks.onCloseModal || (() => {}),
      category: 'navigation'
    },
    
    // Data shortcuts
    {
      key: 'cmd+r',
      description: 'Refresh data',
      action: callbacks.onRefresh || (() => {}),
      category: 'data'
    },
    {
      key: 'cmd+e',
      description: 'Export current view',
      action: callbacks.onExport || (() => {}),
      category: 'data'
    },
    {
      key: 'cmd+s',
      description: 'Save current state',
      action: callbacks.onSave || (() => {}),
      category: 'editing'
    },
    
    // Editing shortcuts
    {
      key: 'cmd+z',
      description: 'Undo last action',
      action: callbacks.onUndo || (() => {}),
      category: 'editing'
    },
    {
      key: 'cmd+shift+z',
      description: 'Redo last action',
      action: callbacks.onRedo || (() => {}),
      category: 'editing'
    },
    
    // View shortcuts
    {
      key: 'cmd+=',
      description: 'Zoom in',
      action: callbacks.onZoomIn || (() => {}),
      category: 'view'
    },
    {
      key: 'cmd+-',
      description: 'Zoom out',
      action: callbacks.onZoomOut || (() => {}),
      category: 'view'
    },
    {
      key: 'f11',
      description: 'Toggle fullscreen',
      action: callbacks.onFullscreen || (() => {}),
      category: 'view'
    },
    
    // Quick access shortcuts
    {
      key: 'cmd+shift+n',
      description: 'Open notifications',
      action: callbacks.onNotifications || (() => {}),
      category: 'navigation'
    },
    {
      key: 'cmd+,',
      description: 'Open settings',
      action: callbacks.onSettings || (() => {}),
      category: 'navigation'
    }
  ];

  return useKeyboardShortcuts({ shortcuts });
};

export default useKeyboardShortcuts;