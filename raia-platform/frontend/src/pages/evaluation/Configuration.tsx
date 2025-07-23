import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Grid,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  MenuItem,
  Alert,
} from '@mui/material';
import { Add, Edit, Delete, Save, Cancel } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../../services/api';
import type { Configuration } from '../../types/evaluation';

const Configuration: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState<Configuration | null>(null);
  const [formData, setFormData] = useState({
    name: '',
    model_name: '',
    temperature: 0.7,
    max_tokens: 1000,
    system_prompt: '',
  });

  const queryClient = useQueryClient();

  const { data: configurations = [], isLoading } = useQuery<Configuration[]>({
    queryKey: ['configurations'],
    queryFn: () => api.getConfigurations(),
  });

  const createMutation = useMutation({
    mutationFn: (data: Omit<Configuration, 'id' | 'created_at' | 'updated_at'>) =>
      api.createConfiguration(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['configurations'] });
      handleClose();
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, ...data }: Configuration) =>
      api.updateConfiguration(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['configurations'] });
      handleClose();
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteConfiguration(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['configurations'] });
    },
  });

  const handleOpen = (config?: Configuration) => {
    if (config) {
      setEditingConfig(config);
      setFormData({
        name: config.name,
        model_name: config.model_name,
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        system_prompt: config.system_prompt,
      });
    } else {
      setEditingConfig(null);
      setFormData({
        name: '',
        model_name: '',
        temperature: 0.7,
        max_tokens: 1000,
        system_prompt: '',
      });
    }
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    setEditingConfig(null);
  };

  const handleSubmit = () => {
    if (editingConfig) {
      updateMutation.mutate({ ...editingConfig, ...formData });
    } else {
      createMutation.mutate(formData);
    }
  };

  const handleDelete = (id: string) => {
    if (window.confirm('Are you sure you want to delete this configuration?')) {
      deleteMutation.mutate(id);
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Configurations</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => handleOpen()}
        >
          New Configuration
        </Button>
      </Box>

      <Grid container spacing={3}>
        {configurations.map((config) => (
          <Grid item xs={12} md={6} lg={4} key={config.id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Typography variant="h6">{config.name}</Typography>
                  <Box>
                    <IconButton
                      size="small"
                      onClick={() => handleOpen(config)}
                      color="primary"
                    >
                      <Edit />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => handleDelete(config.id)}
                      color="error"
                    >
                      <Delete />
                    </IconButton>
                  </Box>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Chip label={config.model_name} color="primary" size="small" sx={{ mr: 1 }} />
                  <Chip label={`Temp: ${config.temperature}`} variant="outlined" size="small" />
                </Box>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Max Tokens: {config.max_tokens}
                </Typography>

                <Typography variant="body2" color="text.secondary" noWrap>
                  {config.system_prompt}
                </Typography>

                <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                  Created: {new Date(config.created_at).toLocaleDateString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingConfig ? 'Edit Configuration' : 'New Configuration'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                select
                label="Model Name"
                value={formData.model_name}
                onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
              >
                <MenuItem value="gpt-4">GPT-4</MenuItem>
                <MenuItem value="gpt-3.5-turbo">GPT-3.5 Turbo</MenuItem>
                <MenuItem value="claude-3-opus">Claude 3 Opus</MenuItem>
                <MenuItem value="claude-3-sonnet">Claude 3 Sonnet</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Temperature"
                value={formData.temperature}
                onChange={(e) => setFormData({ ...formData, temperature: parseFloat(e.target.value) })}
                inputProps={{ min: 0, max: 2, step: 0.1 }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                type="number"
                label="Max Tokens"
                value={formData.max_tokens}
                onChange={(e) => setFormData({ ...formData, max_tokens: parseInt(e.target.value) })}
                inputProps={{ min: 1, max: 8000 }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="System Prompt"
                value={formData.system_prompt}
                onChange={(e) => setFormData({ ...formData, system_prompt: e.target.value })}
              />
            </Grid>
          </Grid>
          
          {(createMutation.error || updateMutation.error) && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {createMutation.error?.message || updateMutation.error?.message}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose} startIcon={<Cancel />}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            variant="contained"
            startIcon={<Save />}
            disabled={createMutation.isPending || updateMutation.isPending}
          >
            {editingConfig ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Configuration;
