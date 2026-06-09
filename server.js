const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const compression = require('compression');
const path = require('path');
require('dotenv').config();

const app = express();

// High-speed production optimization middleware
app.use(helmet({ contentSecurityPolicy: false })); // Permissive CSP for dynamic 3D assets/scripts
app.use(cors());
app.use(compression());
app.use(morgan('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve the frontend interface components directly
app.use(express.static(path.join(__dirname, 'public')));

// Core Commerce API System Check Endpoints
app.get('/api/health', (req, res) => {
  res.status(200).json({ status: "ONLINE", system: "GHOST-3DPOD-ORCHESTRATOR", timestamp: new Date() });
});

// Fallback index delivery route for deep spatial client path components
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Ghost Agentic Core Interface fully responsive on port: ${PORT}`);
});