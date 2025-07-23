import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  Typography,
  Chip,
  Grid,
  Paper,
  IconButton,
  Divider,
  Box
} from '@mui/material';
import {
  X,
  Zap,
  Navigation,
  Database,
  Edit3,
  Eye,
  HelpCircle
} from 'lucide-react';

interface Shortcut {
  key: string;
  description: string;
  category: 'navigation' | 'data' | 'editing' | 'view' | 'help';
}

interface KeyboardShortcutsModalProps {
  open: boolean;
  onClose: () => void;
}

const KeyboardShortcutsModal: React.FC<KeyboardShortcutsModalProps> = ({
  open,
  onClose
}) => {
  const shortcuts: Shortcut[] = [
    // Navigation
    { key: '⌘ + K', description: 'Open quick search', category: 'navigation' },
    { key: '⌘ + B', description: 'Toggle sidebar', category: 'navigation' },
    { key: 'Esc', description: 'Close modal/dialog', category: 'navigation' },
    { key: '⌘ + ⇧ + N', description: 'Open notifications', category: 'navigation' },
    { key: '⌘ + ,', description: 'Open settings', category: 'navigation' },
    
    // Data
    { key: '⌘ + R', description: 'Refresh data', category: 'data' },
    { key: '⌘ + E', description: 'Export current view', category: 'data' },
    { key: '⌘ + S', description: 'Save current state', category: 'data' },
    
    // Editing
    { key: '⌘ + Z', description: 'Undo last action', category: 'editing' },
    { key: '⌘ + ⇧ + Z', description: 'Redo last action', category: 'editing' },
    { key: '⌘ + A', description: 'Select all', category: 'editing' },
    
    // View
    { key: '⌘ + D', description: 'Duplicate current view', category: 'view' },
    { key: '⌘ + +', description: 'Zoom in', category: 'view' },
    { key: '⌘ + -', description: 'Zoom out', category: 'view' },
    { key: 'F11', description: 'Toggle fullscreen', category: 'view' },
    
    // Help
    { key: '⌘ + /', description: 'Show this help', category: 'help' },
  ];

  const categoryIcons = {
    navigation: <Navigation size={20} className="text-blue-500" />,
    data: <Database size={20} className="text-green-500" />,
    editing: <Edit3 size={20} className="text-orange-500" />,
    view: <Eye size={20} className="text-purple-500" />,
    help: <HelpCircle size={20} className="text-indigo-500" />
  };

  const categoryColors = {
    navigation: 'text-blue-600 dark:text-blue-400',
    data: 'text-green-600 dark:text-green-400', 
    editing: 'text-orange-600 dark:text-orange-400',
    view: 'text-purple-600 dark:text-purple-400',
    help: 'text-indigo-600 dark:text-indigo-400'
  };

  const groupedShortcuts = shortcuts.reduce((acc, shortcut) => {
    if (!acc[shortcut.category]) {
      acc[shortcut.category] = [];
    }
    acc[shortcut.category].push(shortcut);
    return acc;
  }, {} as Record<string, Shortcut[]>);

  const formatKey = (key: string) => {
    return key
      .replace('⌘', 'Cmd')
      .replace('⇧', 'Shift')
      .replace('+', ' + ')
      .split(' + ')
      .map((k, index) => (
        <Chip
          key={index}
          label={k.trim()}
          size="small"
          variant="outlined"
          className="mx-0.5 font-mono text-xs"
          sx={{
            height: '24px',
            fontSize: '0.75rem',
            fontFamily: 'monospace',
            borderColor: 'rgba(156, 163, 175, 0.5)',
            backgroundColor: 'rgba(249, 250, 251, 0.8)',
            '&:hover': {
              backgroundColor: 'rgba(243, 244, 246, 1)'
            }
          }}
        />
      ));
  };

  const categoryOrder: Array<keyof typeof groupedShortcuts> = [
    'navigation', 'data', 'editing', 'view', 'help'
  ];

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        className: 'glass rounded-xl border border-white/20 max-h-[80vh]'
      }}
    >
      <DialogTitle className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-3">
          <motion.div
            animate={{ rotate: [0, 360] }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          >
            <Zap size={24} className="text-blue-500" />
          </motion.div>
          <div>
            <Typography variant="h5" className="font-semibold">
              Keyboard Shortcuts
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Master these shortcuts to boost your productivity
            </Typography>
          </div>
        </div>
        <IconButton onClick={onClose} size="small">
          <X size={20} />
        </IconButton>
      </DialogTitle>

      <DialogContent className="p-6 overflow-y-auto custom-scrollbar">
        <Grid container spacing={3}>
          {categoryOrder.map((category) => {
            const categoryShortcuts = groupedShortcuts[category];
            if (!categoryShortcuts) return null;

            return (
              <Grid item xs={12} md={6} key={category}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <Paper 
                    elevation={0}
                    className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-gradient-to-br from-white/50 to-gray-50/30 dark:from-gray-800/50 dark:to-gray-900/30"
                  >
                    <div className="flex items-center space-x-2 mb-3">
                      {categoryIcons[category]}
                      <Typography 
                        variant="h6" 
                        className={`font-medium capitalize ${categoryColors[category]}`}
                      >
                        {category}
                      </Typography>
                    </div>
                    <Divider className="mb-3" />
                    
                    <div className="space-y-2">
                      {categoryShortcuts.map((shortcut, index) => (
                        <motion.div
                          key={shortcut.key}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.05 }}
                          className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                        >
                          <Typography variant="body2" className="flex-1">
                            {shortcut.description}
                          </Typography>
                          <div className="flex items-center space-x-1">
                            {formatKey(shortcut.key)}
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </Paper>
                </motion.div>
              </Grid>
            );
          })}
        </Grid>

        {/* Pro Tips Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-6"
        >
          <Paper 
            elevation={0}
            className="p-4 border border-blue-200 dark:border-blue-800 rounded-lg bg-gradient-to-r from-blue-50/50 to-indigo-50/50 dark:from-blue-900/20 dark:to-indigo-900/20"
          >
            <div className="flex items-center space-x-2 mb-3">
              <Zap size={18} className="text-blue-500" />
              <Typography variant="h6" className="font-medium text-blue-600 dark:text-blue-400">
                Pro Tips
              </Typography>
            </div>
            <div className="space-y-2">
              <Typography variant="body2" color="textSecondary">
                • Hold <Chip label="Cmd" size="small" variant="outlined" className="mx-1" /> and click multiple items to multi-select
              </Typography>
              <Typography variant="body2" color="textSecondary">
                • Use <Chip label="Cmd + K" size="small" variant="outlined" className="mx-1" /> to quickly navigate anywhere
              </Typography>
              <Typography variant="body2" color="textSecondary">
                • Press <Chip label="?" size="small" variant="outlined" className="mx-1" /> on any page for contextual help
              </Typography>
            </div>
          </Paper>
        </motion.div>

        {/* Footer */}
        <Box className="mt-6 text-center">
          <Typography variant="caption" color="textSecondary">
            Shortcuts work globally except when typing in input fields
          </Typography>
        </Box>
      </DialogContent>
    </Dialog>
  );
};

export default KeyboardShortcutsModal;