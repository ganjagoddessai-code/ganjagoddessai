const { Sequelize } = require('sequelize');
const mongoose = require('mongoose');
const Redis = require('ioredis');
require('dotenv').config();

// 1. Relational Database Initialization (PostgreSQL)
const sequelize = new Sequelize(process.env.DATABASE_URL, {
  dialect: 'postgres',
  logging: false,
  pool: { max: 10, min: 0, acquire: 30000, idle: 10000 }
});

// 2. Document Cache Strategy Store Connection (MongoDB)
const connectMongo = async () => {
  try {
    await mongoose.connect(process.env.MONGO_URI);
    console.log("💾 MongoDB Cluster Connection Secure.");
  } catch (err) {
    console.error("❌ MongoDB Database drop exception:", err.message);
  }
};

// 3. High-Speed Task Matrix Memory Store Connection (Redis)
const redis = new Redis(process.env.REDIS_URL);

module.exports = { sequelize, connectMongo, redis };