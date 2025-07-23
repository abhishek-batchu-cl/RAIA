import React from 'react';
import { ChevronRight, Home, ArrowLeft } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '../../utils';

interface BreadcrumbItem {
  id: string;
  label: string;
  description?: string;
  icon?: React.ComponentType<any>;
}

interface BreadcrumbProps {
  items: BreadcrumbItem[];
  onNavigate: (itemId: string) => void;
  onBack?: () => void;
  showBackButton?: boolean;
  className?: string;
}

const Breadcrumb: React.FC<BreadcrumbProps> = ({
  items,
  onNavigate,
  onBack,
  showBackButton = false,
  className = '',
}) => {
  const currentItem = items[items.length - 1];

  return (
    <div className={cn("flex items-center space-x-2", className)}>
      {/* Back Button */}
      {showBackButton && onBack && (
        <button
          onClick={onBack}
          className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
          title="Go back"
        >
          <ArrowLeft className="w-4 h-4 text-neutral-600 dark:text-neutral-400" />
        </button>
      )}

      {/* Breadcrumb Items */}
      <div className="flex items-center space-x-1 overflow-hidden">
        {items.map((item, index) => {
          const isLast = index === items.length - 1;
          const Icon = item.icon;
          
          return (
            <React.Fragment key={item.id}>
              {index > 0 && (
                <ChevronRight className="w-4 h-4 text-neutral-400 flex-shrink-0" />
              )}
              
              <motion.button
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => !isLast && onNavigate(item.id)}
                disabled={isLast}
                className={cn(
                  "flex items-center space-x-2 px-2 py-1 rounded-md transition-colors truncate",
                  isLast
                    ? "text-neutral-900 dark:text-white font-medium cursor-default"
                    : "text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-100 dark:hover:bg-neutral-800 cursor-pointer"
                )}
                title={item.description}
              >
                {Icon && index === 0 && (
                  <Icon className="w-4 h-4 flex-shrink-0" />
                )}
                <span className="truncate text-sm">{item.label}</span>
              </motion.button>
            </React.Fragment>
          );
        })}
      </div>

      {/* Current Page Description */}
      {currentItem.description && (
        <div className="hidden md:flex items-center space-x-2 pl-4 border-l border-neutral-200 dark:border-neutral-700">
          <span className="text-xs text-neutral-500 dark:text-neutral-400 truncate">
            {currentItem.description}
          </span>
        </div>
      )}
    </div>
  );
};

export default Breadcrumb;