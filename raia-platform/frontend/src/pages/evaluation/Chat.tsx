import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Paper,
  List,
  ListItem,
  ListItemText,
  Divider,
  Chip,
  CircularProgress,
} from '@mui/material';
import { Send, Person, SmartToy } from '@mui/icons-material';
import { useMutation } from '@tanstack/react-query';
import { api } from '../../services/api';
import type { ChatMessage, ChatResponse } from '../../types/evaluation';

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');

  const chatMutation = useMutation({
    mutationFn: (message: string) => api.chat(message),
    onSuccess: (response: ChatResponse) => {
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.response,
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, assistantMessage]);
    },
  });

  const handleSend = () => {
    if (!input.trim()) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    chatMutation.mutate(input);
    setInput('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Chat with RAG Agent
      </Typography>
      
      <Card sx={{ height: 'calc(100vh - 200px)', display: 'flex', flexDirection: 'column' }}>
        {/* Messages Area */}
        <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
          {messages.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="body1" color="text.secondary">
                Start a conversation with the RAG agent. Ask questions about the uploaded documents.
              </Typography>
            </Box>
          ) : (
            <List>
              {messages.map((message, index) => (
                <ListItem key={index} sx={{ flexDirection: 'column', alignItems: 'flex-start', px: 0 }}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 2,
                      width: '100%',
                      mb: 1,
                    }}
                  >
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: 40,
                        height: 40,
                        borderRadius: '50%',
                        backgroundColor: message.role === 'user' ? 'primary.main' : 'secondary.main',
                        color: 'white',
                      }}
                    >
                      {message.role === 'user' ? <Person /> : <SmartToy />}
                    </Box>
                    
                    <Box sx={{ flexGrow: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {message.role === 'user' ? 'You' : 'RAG Agent'}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </Typography>
                      </Box>
                      
                      <Paper
                        variant="outlined"
                        sx={{
                          p: 2,
                          backgroundColor: message.role === 'user' ? 'primary.50' : 'grey.50',
                          borderColor: message.role === 'user' ? 'primary.200' : 'grey.200',
                        }}
                      >
                        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                          {message.content}
                        </Typography>
                      </Paper>
                    </Box>
                  </Box>
                  
                  {index < messages.length - 1 && <Divider sx={{ width: '100%', mt: 2 }} />}
                </ListItem>
              ))}
              
              {chatMutation.isPending && (
                <ListItem sx={{ justifyContent: 'center' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <CircularProgress size={20} />
                    <Typography variant="body2" color="text.secondary">
                      RAG Agent is thinking...
                    </Typography>
                  </Box>
                </ListItem>
              )}
            </List>
          )}
        </Box>

        {/* Input Area */}
        <Divider />
        <Box sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <TextField
              fullWidth
              multiline
              maxRows={4}
              placeholder="Ask a question about the documents..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={chatMutation.isPending}
            />
            <Button
              variant="contained"
              endIcon={<Send />}
              onClick={handleSend}
              disabled={!input.trim() || chatMutation.isPending}
              sx={{ minWidth: 100 }}
            >
              Send
            </Button>
          </Box>
          
          {chatMutation.error && (
            <Typography variant="body2" color="error" sx={{ mt: 1 }}>
              Error: {chatMutation.error.message}
            </Typography>
          )}
        </Box>
      </Card>

      {/* Quick Actions */}
      <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <Typography variant="body2" color="text.secondary" sx={{ mr: 2 }}>
          Quick questions:
        </Typography>
        {[
          'What documents are available?',
          'Summarize the main topics',
          'What are the key findings?',
        ].map((question) => (
          <Chip
            key={question}
            label={question}
            variant="outlined"
            clickable
            onClick={() => setInput(question)}
            disabled={chatMutation.isPending}
          />
        ))}
      </Box>
    </Box>
  );
};

export default Chat;
