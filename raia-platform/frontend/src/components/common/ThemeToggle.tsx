import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  Switch,
  FormControlLabel,
  Paper,
  Chip
} from '@mui/material';
import {
  Sun,
  Moon,
  Monitor,
  Palette,
  Settings,
  Clock,
  Eye,
  Zap
} from 'lucide-react';

type ThemeMode = 'light' | 'dark' | 'auto';
type CustomTheme = 'default' | 'midnight' | 'sunset' | 'ocean' | 'forest';

interface ThemeToggleProps {
  currentTheme?: ThemeMode;
  onThemeChange?: (theme: ThemeMode) => void;
  showCustomThemes?: boolean;
  showScheduling?: boolean;
  showAccessibility?: boolean;
  className?: string;
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({
  currentTheme = 'auto',
  onThemeChange,
  showCustomThemes = true,
  showScheduling = true, 
  showAccessibility = true,
  className = ''
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedTheme, setSelectedTheme] = useState<ThemeMode>(currentTheme);
  const [customTheme, setCustomTheme] = useState<CustomTheme>('default');
  const [autoSchedule, setAutoSchedule] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const [systemTheme, setSystemTheme] = useState<'light' | 'dark'>('light');

  // Detect system theme preference
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    setSystemTheme(mediaQuery.matches ? 'dark' : 'light');

    const handler = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? 'dark' : 'light');
    };

    mediaQuery.addListener(handler);
    return () => mediaQuery.removeListener(handler);
  }, []);

  // Apply theme
  useEffect(() => {
    const effectiveTheme = selectedTheme === 'auto' ? systemTheme : selectedTheme;
    document.documentElement.setAttribute('data-theme', effectiveTheme);
    document.documentElement.classList.toggle('dark', effectiveTheme === 'dark');
    
    if (highContrast) {
      document.documentElement.classList.add('high-contrast');
    } else {
      document.documentElement.classList.remove('high-contrast');
    }
  }, [selectedTheme, systemTheme, highContrast]);

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleThemeChange = (theme: ThemeMode) => {
    setSelectedTheme(theme);
    onThemeChange?.(theme);
    handleClose();
  };

  const getThemeIcon = (theme: ThemeMode) => {
    switch (theme) {
      case 'light':
        return <Sun size={18} className="text-yellow-500" />;
      case 'dark':
        return <Moon size={18} className="text-blue-400" />;
      case 'auto':
        return <Monitor size={18} className="text-gray-500" />;
    }
  };

  const getCurrentIcon = () => {
    const effectiveTheme = selectedTheme === 'auto' ? systemTheme : selectedTheme;
    return getThemeIcon(effectiveTheme);
  };

  const customThemes = [
    { id: 'default', name: 'Default', color: '#3B82F6' },
    { id: 'midnight', name: 'Midnight', color: '#1E1B4B' },
    { id: 'sunset', name: 'Sunset', color: '#F59E0B' },
    { id: 'ocean', name: 'Ocean', color: '#0891B2' },
    { id: 'forest', name: 'Forest', color: '#059669' }
  ];

  const isOpen = Boolean(anchorEl);

  return (
    <div className={className}>
      <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
        <IconButton
          onClick={handleClick}
          className="relative"
          size="medium"
          aria-label="Toggle theme"
        >
          <motion.div
            key={selectedTheme}
            initial={{ rotate: -180, opacity: 0 }}
            animate={{ rotate: 0, opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            {getCurrentIcon()}
          </motion.div>
          
          {selectedTheme === 'auto' && (
            <motion.div
              className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white dark:border-gray-800"
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          )}
        </IconButton>
      </motion.div>

      <Menu
        anchorEl={anchorEl}
        open={isOpen}
        onClose={handleClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        PaperProps={{
          className: 'glass rounded-xl border border-white/20 min-w-[280px]'
        }}
      >
        <div className="p-3 border-b border-gray-200 dark:border-gray-700">
          <Typography variant="subtitle1" className="font-semibold flex items-center">
            <Palette size={18} className="mr-2" />
            Theme Settings
          </Typography>
          <Typography variant="caption" color="textSecondary">
            Customize your visual experience
          </Typography>
        </div>

        {/* Theme Mode Selection */}
        <div className="p-2">
          <Typography variant="caption" className="px-3 py-1 text-gray-600 dark:text-gray-400 uppercase tracking-wide">
            Theme Mode
          </Typography>
          
          {(['light', 'dark', 'auto'] as ThemeMode[]).map((theme) => (
            <MenuItem
              key={theme}
              selected={selectedTheme === theme}
              onClick={() => handleThemeChange(theme)}
              className="rounded-lg mx-2 my-1"
            >
              <ListItemIcon>
                {getThemeIcon(theme)}
              </ListItemIcon>
              <ListItemText
                primary={
                  <div className="flex items-center justify-between">
                    <span className="capitalize">{theme}</span>
                    {theme === 'auto' && (
                      <Chip
                        size="small"
                        label={`${systemTheme}`}
                        variant="outlined"
                        className="ml-2 text-xs"
                      />
                    )}
                  </div>
                }
                secondary={
                  theme === 'auto' ? 'Follows system preference' :
                  theme === 'light' ? 'Always light theme' :
                  'Always dark theme'
                }
              />
            </MenuItem>
          ))}
        </div>

        <Divider />

        {/* Custom Themes */}
        {showCustomThemes && (
          <div className="p-2">
            <Typography variant="caption" className="px-3 py-1 text-gray-600 dark:text-gray-400 uppercase tracking-wide">
              Custom Themes
            </Typography>
            <div className="grid grid-cols-5 gap-2 p-3">
              {customThemes.map((theme) => (
                <motion.button
                  key={theme.id}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setCustomTheme(theme.id as CustomTheme)}
                  className={`w-8 h-8 rounded-full border-2 relative ${
                    customTheme === theme.id 
                      ? 'border-blue-500 ring-2 ring-blue-200 dark:ring-blue-800' 
                      : 'border-gray-300 dark:border-gray-600'
                  }`}
                  style={{ backgroundColor: theme.color }}
                  title={theme.name}
                >
                  {customTheme === theme.id && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="absolute inset-0 flex items-center justify-center"
                    >
                      <div className="w-2 h-2 bg-white rounded-full" />
                    </motion.div>
                  )}
                </motion.button>
              ))}
            </div>
          </div>
        )}

        {showCustomThemes && <Divider />}

        {/* Scheduling */}
        {showScheduling && (
          <div className="p-2">
            <MenuItem className="rounded-lg mx-2 my-1">
              <ListItemIcon>
                <Clock size={18} />
              </ListItemIcon>
              <ListItemText primary="Auto Schedule" />
              <Switch
                checked={autoSchedule}
                onChange={(e) => setAutoSchedule(e.target.checked)}
                size="small"
              />
            </MenuItem>
            
            {autoSchedule && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="px-3 pb-2"
              >
                <Paper elevation={0} className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <Typography variant="caption" color="textSecondary">
                    <Zap size={14} className="inline mr-1" />
                    Light mode: 7:00 AM - 7:00 PM<br />
                    Dark mode: 7:00 PM - 7:00 AM
                  </Typography>
                </Paper>
              </motion.div>
            )}
          </div>
        )}

        {showScheduling && <Divider />}

        {/* Accessibility */}
        {showAccessibility && (
          <div className="p-2">
            <Typography variant="caption" className="px-3 py-1 text-gray-600 dark:text-gray-400 uppercase tracking-wide">
              Accessibility
            </Typography>
            
            <MenuItem className="rounded-lg mx-2 my-1">
              <ListItemIcon>
                <Eye size={18} />
              </ListItemIcon>
              <ListItemText 
                primary="High Contrast"
                secondary="Increase color contrast for better visibility"
              />
              <Switch
                checked={highContrast}
                onChange={(e) => setHighContrast(e.target.checked)}
                size="small"
              />
            </MenuItem>
          </div>
        )}

        <Divider />

        {/* Settings Link */}
        <div className="p-2">
          <MenuItem className="rounded-lg mx-2 my-1 text-gray-600 dark:text-gray-400">
            <ListItemIcon>
              <Settings size={18} />
            </ListItemIcon>
            <ListItemText primary="More theme settings..." />
          </MenuItem>
        </div>
      </Menu>
    </div>
  );
};

export default ThemeToggle;