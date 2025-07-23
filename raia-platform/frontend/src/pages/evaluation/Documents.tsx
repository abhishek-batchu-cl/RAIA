import React, { useState, useRef } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Alert,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Upload,
  Delete,
  Visibility,
  CloudUpload,
  Description,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../../services/api';
import type { Document } from '../../types/evaluation';

const Documents: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewDialog, setPreviewDialog] = useState<{ open: boolean; document: Document | null }>({
    open: false,
    document: null,
  });
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const queryClient = useQueryClient();

  const { data: documents = [], isLoading } = useQuery<Document[]>({
    queryKey: ['documents'],
    queryFn: () => api.getDocuments(),
  });

  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.uploadDocument(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (documentId: string) => api.deleteDocument(documentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleUpload = () => {
    if (selectedFile) {
      uploadMutation.mutate(selectedFile);
    }
  };

  const handleDelete = (documentId: string, filename: string) => {
    if (window.confirm(`Are you sure you want to delete "${filename}"?`)) {
      deleteMutation.mutate(documentId);
    }
  };

  const handlePreview = (document: Document) => {
    setPreviewDialog({ open: true, document });
  };

  const handleClosePreview = () => {
    setPreviewDialog({ open: false, document: null });
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getFileIcon = (filename: string) => {
    const extension = filename.split('.').pop()?.toLowerCase();
    return <Description color="primary" />;
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Document Management
      </Typography>

      <Grid container spacing={3}>
        {/* Upload Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upload Documents
              </Typography>
              
              <Box
                sx={{
                  border: 2,
                  borderColor: selectedFile ? 'primary.main' : 'grey.300',
                  borderStyle: 'dashed',
                  borderRadius: 2,
                  p: 3,
                  textAlign: 'center',
                  backgroundColor: selectedFile ? 'primary.50' : 'grey.50',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease-in-out',
                  '&:hover': {
                    borderColor: 'primary.main',
                    backgroundColor: 'primary.50',
                  },
                }}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileSelect}
                  accept=".txt,.pdf,.docx,.md"
                  style={{ display: 'none' }}
                />
                
                <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                
                {selectedFile ? (
                  <Box>
                    <Typography variant="h6" color="primary">
                      {selectedFile.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {formatFileSize(selectedFile.size)}
                    </Typography>
                  </Box>
                ) : (
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Drop files here or click to browse
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Supported formats: .txt, .pdf, .docx, .md
                    </Typography>
                  </Box>
                )}
              </Box>

              {selectedFile && (
                <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                  <Button
                    variant="contained"
                    startIcon={<Upload />}
                    onClick={handleUpload}
                    disabled={uploadMutation.isPending}
                    fullWidth
                  >
                    {uploadMutation.isPending ? 'Uploading...' : 'Upload Document'}
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={() => {
                      setSelectedFile(null);
                      if (fileInputRef.current) {
                        fileInputRef.current.value = '';
                      }
                    }}
                  >
                    Cancel
                  </Button>
                </Box>
              )}

              {uploadMutation.error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {uploadMutation.error.message}
                </Alert>
              )}

              {uploadMutation.isSuccess && (
                <Alert severity="success" sx={{ mt: 2 }}>
                  Document uploaded successfully!
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Document Stats */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Document Library
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Box>
                  <Typography variant="h4" color="primary">
                    {documents.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Documents
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="h4" color="secondary">
                    {documents.reduce((acc, doc) => acc + (doc.metadata?.size || 0), 0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Size (bytes)
                  </Typography>
                </Box>
              </Box>

              <Typography variant="body2" color="text.secondary">
                Documents are processed and indexed automatically for RAG capabilities.
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Documents Table */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Uploaded Documents
              </Typography>
              
              {isLoading ? (
                <LinearProgress />
              ) : documents.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body1" color="text.secondary">
                    No documents uploaded yet. Upload your first document to get started.
                  </Typography>
                </Box>
              ) : (
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>File</TableCell>
                        <TableCell>Size</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Uploaded</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {documents.map((document) => (
                        <TableRow key={document.id}>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              {getFileIcon(document.filename)}
                              <Typography variant="body2">
                                {document.filename}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell>
                            {formatFileSize(document.metadata?.size || 0)}
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={document.filename.split('.').pop()?.toUpperCase() || 'Unknown'}
                              size="small"
                              color="primary"
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>
                            {new Date(document.uploaded_at).toLocaleDateString()}
                          </TableCell>
                          <TableCell>
                            <Chip
                              label="Processed"
                              size="small"
                              color="success"
                            />
                          </TableCell>
                          <TableCell>
                            <Box sx={{ display: 'flex', gap: 1 }}>
                              <IconButton
                                size="small"
                                color="primary"
                                onClick={() => handlePreview(document)}
                                title="Preview"
                              >
                                <Visibility />
                              </IconButton>
                              <IconButton
                                size="small"
                                color="error"
                                onClick={() => handleDelete(document.id, document.filename)}
                                title="Delete"
                              >
                                <Delete />
                              </IconButton>
                            </Box>
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

      {/* Preview Dialog */}
      <Dialog
        open={previewDialog.open}
        onClose={handleClosePreview}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {previewDialog.document?.filename}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
              {previewDialog.document?.content || 'No content available'}
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClosePreview}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Documents;
