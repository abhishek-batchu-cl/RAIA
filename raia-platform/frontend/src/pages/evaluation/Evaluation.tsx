import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Grid,
  MenuItem,
  TextField,
  LinearProgress,
  Alert,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import { PlayArrow, Stop, Refresh } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../../services/api';
import type { Configuration, EvaluationResult } from '../../types/evaluation';

const Evaluation: React.FC = () => {
  const [selectedConfig, setSelectedConfig] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  
  const queryClient = useQueryClient();

  const { data: configurations = [] } = useQuery<Configuration[]>({
    queryKey: ['configurations'],
    queryFn: () => api.getConfigurations(),
  });

  const { data: datasets = [] } = useQuery<string[]>({
    queryKey: ['datasets'],
    queryFn: () => api.getDatasets(),
  });

  const { data: evaluations = [], isLoading } = useQuery<EvaluationResult[]>({
    queryKey: ['evaluations'],
    queryFn: () => api.getEvaluations(),
    refetchInterval: 5000, // Refresh every 5 seconds for running evaluations
  });

  const startEvaluationMutation = useMutation({
    mutationFn: ({ configId, dataset }: { configId: string; dataset: string }) =>
      api.startEvaluation(configId, dataset),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['evaluations'] });
    },
  });

  const stopEvaluationMutation = useMutation({
    mutationFn: (evaluationId: string) => api.stopEvaluation(evaluationId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['evaluations'] });
    },
  });

  const handleStartEvaluation = () => {
    if (selectedConfig && selectedDataset) {
      startEvaluationMutation.mutate({
        configId: selectedConfig,
        dataset: selectedDataset,
      });
      setSelectedConfig('');
      setSelectedDataset('');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'warning';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatScore = (score: number) => (score * 100).toFixed(1) + '%';

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Evaluation
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Start New Evaluation
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    select
                    label="Configuration"
                    value={selectedConfig}
                    onChange={(e) => setSelectedConfig(e.target.value)}
                  >
                    {configurations.map((config) => (
                      <MenuItem key={config.id} value={config.id}>
                        {config.name} ({config.model_name})
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    select
                    label="Dataset"
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                  >
                    {datasets.map((dataset) => (
                      <MenuItem key={dataset} value={dataset}>
                        {dataset}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                
                <Grid item xs={12}>
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<PlayArrow />}
                    onClick={handleStartEvaluation}
                    disabled={
                      !selectedConfig ||
                      !selectedDataset ||
                      startEvaluationMutation.isPending
                    }
                  >
                    Start Evaluation
                  </Button>
                </Grid>
              </Grid>

              {startEvaluationMutation.error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {startEvaluationMutation.error.message}
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Evaluation Summary
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Chip
                  label={`Total: ${evaluations.length}`}
                  color="primary"
                  variant="outlined"
                />
                <Chip
                  label={`Running: ${evaluations.filter(e => e.status === 'running').length}`}
                  color="warning"
                  variant="outlined"
                />
                <Chip
                  label={`Completed: ${evaluations.filter(e => e.status === 'completed').length}`}
                  color="success"
                  variant="outlined"
                />
                <Chip
                  label={`Failed: ${evaluations.filter(e => e.status === 'failed').length}`}
                  color="error"
                  variant="outlined"
                />
              </Box>

              <Button
                startIcon={<Refresh />}
                onClick={() => queryClient.invalidateQueries({ queryKey: ['evaluations'] })}
                sx={{ mt: 2 }}
              >
                Refresh
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Evaluation Results
              </Typography>
              
              {isLoading ? (
                <LinearProgress />
              ) : (
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Configuration</TableCell>
                        <TableCell>Dataset</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Overall Score</TableCell>
                        <TableCell>Accuracy</TableCell>
                        <TableCell>Relevance</TableCell>
                        <TableCell>Started</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {evaluations.map((evaluation) => (
                        <TableRow key={evaluation.id}>
                          <TableCell>
                            {configurations.find(c => c.id === evaluation.configuration_id)?.name || 'Unknown'}
                          </TableCell>
                          <TableCell>{evaluation.dataset_name}</TableCell>
                          <TableCell>
                            <Chip
                              label={evaluation.status}
                              color={getStatusColor(evaluation.status) as any}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            {evaluation.results ? formatScore(evaluation.results.overall_score) : '-'}
                          </TableCell>
                          <TableCell>
                            {evaluation.results ? formatScore(evaluation.results.accuracy) : '-'}
                          </TableCell>
                          <TableCell>
                            {evaluation.results ? formatScore(evaluation.results.relevance) : '-'}
                          </TableCell>
                          <TableCell>
                            {new Date(evaluation.created_at).toLocaleString()}
                          </TableCell>
                          <TableCell>
                            {evaluation.status === 'running' && (
                              <Button
                                size="small"
                                color="error"
                                startIcon={<Stop />}
                                onClick={() => stopEvaluationMutation.mutate(evaluation.id)}
                              >
                                Stop
                              </Button>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Evaluation;
