import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Checkbox,
  TableSortLabel,
  TextField,
  InputAdornment,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Typography,
  Box
} from '@mui/material';
import {
  Search,
  Filter,
  Download,
  MoreVertical,
  Eye,
  Edit,
  Trash2,
  ArrowUpDown,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';

interface Column {
  id: string;
  label: string;
  minWidth?: number;
  align?: 'right' | 'left' | 'center';
  format?: (value: any) => string;
  sortable?: boolean;
  filterable?: boolean;
  render?: (value: any, row: any) => React.ReactNode;
}

interface EnterpriseDataTableProps {
  columns: Column[];
  rows: any[];
  title?: string;
  searchable?: boolean;
  selectable?: boolean;
  sortable?: boolean;
  filterable?: boolean;
  exportable?: boolean;
  pagination?: boolean;
  rowsPerPage?: number;
  onRowClick?: (row: any) => void;
  onRowSelect?: (selectedRows: any[]) => void;
  onExport?: () => void;
  actions?: Array<{
    label: string;
    icon: React.ReactNode;
    onClick: (row: any) => void;
    color?: 'primary' | 'secondary' | 'error' | 'warning' | 'info' | 'success';
  }>;
  className?: string;
}

const EnterpriseDataTable: React.FC<EnterpriseDataTableProps> = ({
  columns,
  rows,
  title,
  searchable = true,
  selectable = true,
  sortable = true,
  filterable = false,
  exportable = true,
  pagination = true,
  rowsPerPage = 10,
  onRowClick,
  onRowSelect,
  onExport,
  actions = [
    { label: 'View', icon: <Eye size={16} />, onClick: () => {} },
    { label: 'Edit', icon: <Edit size={16} />, onClick: () => {} },
    { label: 'Delete', icon: <Trash2 size={16} />, onClick: () => {}, color: 'error' as const }
  ],
  className = ''
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRows, setSelectedRows] = useState<any[]>([]);
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' } | null>(null);
  const [page, setPage] = useState(0);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedRow, setSelectedRow] = useState<any>(null);

  // Filter and sort data
  const filteredRows = rows.filter(row =>
    searchTerm === '' || 
    columns.some(col => 
      String(row[col.id]).toLowerCase().includes(searchTerm.toLowerCase())
    )
  );

  const sortedRows = React.useMemo(() => {
    if (!sortConfig) return filteredRows;

    return [...filteredRows].sort((a, b) => {
      const aVal = a[sortConfig.key];
      const bVal = b[sortConfig.key];
      
      if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });
  }, [filteredRows, sortConfig]);

  // Pagination
  const paginatedRows = pagination 
    ? sortedRows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
    : sortedRows;

  const handleSort = (columnId: string) => {
    if (!sortable) return;
    
    setSortConfig(current => ({
      key: columnId,
      direction: current?.key === columnId && current.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const handleSelectAll = () => {
    if (selectedRows.length === paginatedRows.length) {
      setSelectedRows([]);
    } else {
      setSelectedRows([...paginatedRows]);
    }
    onRowSelect?.(selectedRows);
  };

  const handleRowSelect = (row: any) => {
    const newSelection = selectedRows.includes(row)
      ? selectedRows.filter(r => r !== row)
      : [...selectedRows, row];
    
    setSelectedRows(newSelection);
    onRowSelect?.(newSelection);
  };

  const handleActionClick = (event: React.MouseEvent<HTMLElement>, row: any) => {
    setAnchorEl(event.currentTarget);
    setSelectedRow(row);
  };

  const renderCellContent = (column: Column, row: any) => {
    const value = row[column.id];
    
    if (column.render) {
      return column.render(value, row);
    }
    
    if (column.format) {
      return column.format(value);
    }
    
    return value;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 ${className}`}
    >
      {/* Header */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          {title && (
            <Typography variant="h6" className="font-semibold text-gray-900 dark:text-gray-100">
              {title}
            </Typography>
          )}
          <div className="flex items-center space-x-2">
            {selectedRows.length > 0 && (
              <Chip
                label={`${selectedRows.length} selected`}
                color="primary"
                size="small"
                onDelete={() => setSelectedRows([])}
              />
            )}
            {exportable && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onExport}
                className="px-3 py-2 bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors flex items-center space-x-2"
              >
                <Download size={16} />
                <span>Export</span>
              </motion.button>
            )}
          </div>
        </div>

        {/* Search and Filters */}
        <div className="flex items-center space-x-4">
          {searchable && (
            <TextField
              placeholder="Search..."
              variant="outlined"
              size="small"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search size={16} className="text-gray-400" />
                  </InputAdornment>
                ),
              }}
              className="flex-1 max-w-md"
            />
          )}
          {filterable && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors flex items-center space-x-2"
            >
              <Filter size={16} />
              <span>Filter</span>
            </motion.button>
          )}
        </div>
      </div>

      {/* Table */}
      <TableContainer>
        <Table stickyHeader>
          <TableHead>
            <TableRow>
              {selectable && (
                <TableCell padding="checkbox">
                  <Checkbox
                    indeterminate={selectedRows.length > 0 && selectedRows.length < paginatedRows.length}
                    checked={paginatedRows.length > 0 && selectedRows.length === paginatedRows.length}
                    onChange={handleSelectAll}
                  />
                </TableCell>
              )}
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.align}
                  style={{ minWidth: column.minWidth }}
                  className="font-semibold bg-gray-50 dark:bg-gray-700"
                >
                  {column.sortable !== false && sortable ? (
                    <TableSortLabel
                      active={sortConfig?.key === column.id}
                      direction={sortConfig?.key === column.id ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort(column.id)}
                      className="hover:text-blue-600"
                    >
                      {column.label}
                    </TableSortLabel>
                  ) : (
                    column.label
                  )}
                </TableCell>
              ))}
              {actions.length > 0 && (
                <TableCell align="center" className="font-semibold bg-gray-50 dark:bg-gray-700">
                  Actions
                </TableCell>
              )}
            </TableRow>
          </TableHead>
          <TableBody>
            <AnimatePresence>
              {paginatedRows.map((row, index) => (
                <motion.tr
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: index * 0.05 }}
                  className="hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer"
                  onClick={() => onRowClick?.(row)}
                >
                  {selectable && (
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={selectedRows.includes(row)}
                        onChange={() => handleRowSelect(row)}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </TableCell>
                  )}
                  {columns.map((column) => (
                    <TableCell key={column.id} align={column.align}>
                      {renderCellContent(column, row)}
                    </TableCell>
                  ))}
                  {actions.length > 0 && (
                    <TableCell align="center">
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleActionClick(e, row);
                        }}
                        className="hover:bg-gray-100 dark:hover:bg-gray-600"
                      >
                        <MoreVertical size={16} />
                      </IconButton>
                    </TableCell>
                  )}
                </motion.tr>
              ))}
            </AnimatePresence>
          </TableBody>
        </Table>
      </TableContainer>

      {/* Pagination */}
      {pagination && (
        <div className="flex items-center justify-between p-4 border-t border-gray-200 dark:border-gray-700">
          <Typography variant="body2" color="textSecondary">
            Showing {page * rowsPerPage + 1} to {Math.min((page + 1) * rowsPerPage, filteredRows.length)} of {filteredRows.length} entries
          </Typography>
          <div className="flex items-center space-x-2">
            <IconButton
              disabled={page === 0}
              onClick={() => setPage(page - 1)}
              size="small"
            >
              <ChevronLeft size={16} />
            </IconButton>
            <Typography variant="body2" className="px-2">
              Page {page + 1} of {Math.ceil(filteredRows.length / rowsPerPage)}
            </Typography>
            <IconButton
              disabled={page >= Math.ceil(filteredRows.length / rowsPerPage) - 1}
              onClick={() => setPage(page + 1)}
              size="small"
            >
              <ChevronRight size={16} />
            </IconButton>
          </div>
        </div>
      )}

      {/* Action Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        {actions.map((action, index) => (
          <MenuItem
            key={index}
            onClick={() => {
              action.onClick(selectedRow);
              setAnchorEl(null);
            }}
            className="flex items-center space-x-2"
          >
            {action.icon}
            <span>{action.label}</span>
          </MenuItem>
        ))}
      </Menu>
    </motion.div>
  );
};

export default EnterpriseDataTable;