import React from 'react';
import { motion } from 'framer-motion';
import { Breadcrumbs, Typography, Link, Chip, Avatar } from '@mui/material';
import { 
  Home, 
  ChevronRight, 
  LayoutDashboard,
  FileSpreadsheet,
  Settings,
  TrendingUp,
  Shield,
  Database,
  Brain,
  AlertTriangle
} from 'lucide-react';

interface BreadcrumbItem {
  label: string;
  path?: string;
  icon?: keyof typeof iconMap;
  active?: boolean;
  badge?: string | number;
  color?: 'default' | 'primary' | 'secondary' | 'success' | 'warning' | 'error';
}

const iconMap = {
  home: Home,
  dashboard: LayoutDashboard,
  assessment: FileSpreadsheet,
  settings: Settings,
  trending: TrendingUp,
  security: Shield,
  data: Database,
  brain: Brain,
  alert: AlertTriangle,
};

interface EnterpriseBreadcrumbsProps {
  items: BreadcrumbItem[];
  showIcons?: boolean;
  showHome?: boolean;
  maxItems?: number;
  className?: string;
}

const EnterpriseBreadcrumbs: React.FC<EnterpriseBreadcrumbsProps> = ({
  items,
  showIcons = true,
  showHome = true,
  maxItems = 5,
  className = ''
}) => {
  const homeItem: BreadcrumbItem = {
    label: 'Home',
    path: '/',
    icon: 'home'
  };

  const allItems = showHome ? [homeItem, ...items] : items;
  const displayItems = allItems.slice(-maxItems);

  const renderBreadcrumbItem = (item: BreadcrumbItem, index: number, isLast: boolean) => {
    const IconComponent = item.icon ? iconMap[item.icon] : null;
    
    const content = (
      <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: index * 0.05 }}
        className="flex items-center space-x-2"
      >
        {showIcons && IconComponent && (
          <IconComponent size={16} className="text-gray-500" />
        )}
        <span className={`font-medium ${
          isLast 
            ? 'text-gray-900 dark:text-gray-100' 
            : 'text-gray-600 dark:text-gray-400'
        }`}>
          {item.label}
        </span>
        {item.badge && (
          <Chip
            size="small"
            label={item.badge}
            color={item.color || 'primary'}
            variant="outlined"
            sx={{ 
              height: 20, 
              fontSize: '0.75rem',
              '& .MuiChip-label': {
                px: 1
              }
            }}
          />
        )}
      </motion.div>
    );

    if (isLast || !item.path) {
      return (
        <Typography 
          key={index}
          color={isLast ? 'textPrimary' : 'textSecondary'}
          className="flex items-center"
        >
          {content}
        </Typography>
      );
    }

    return (
      <Link
        key={index}
        underline="hover"
        color="inherit"
        href={item.path}
        className="flex items-center transition-colors hover:text-blue-600 dark:hover:text-blue-400"
        onClick={(e) => {
          e.preventDefault();
          // Handle navigation - you can implement routing here
          console.log(`Navigate to: ${item.path}`);
        }}
      >
        {content}
      </Link>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-white/50 dark:bg-gray-900/50 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700 px-6 py-3 ${className}`}
    >
      <div className="flex items-center justify-between">
        <Breadcrumbs
          separator={<ChevronRight size={14} className="text-gray-400" />}
          maxItems={maxItems}
          className="text-sm"
        >
          {displayItems.map((item, index) =>
            renderBreadcrumbItem(item, index, index === displayItems.length - 1)
          )}
        </Breadcrumbs>
        
        {/* Quick Actions */}
        <div className="flex items-center space-x-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="text-xs px-3 py-1 bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-full hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors"
          >
            Export
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="text-xs px-3 py-1 bg-green-50 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-full hover:bg-green-100 dark:hover:bg-green-900/50 transition-colors"
          >
            Help
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};

export default EnterpriseBreadcrumbs;