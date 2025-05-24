import express from 'express';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import cors from 'cors';
import bodyParser from 'body-parser';
import contactRoutes from './server/routes/contactRoutes.js';  // Note the '.js' extension

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Routes
app.use('/api/contact', contactRoutes);

async function connectToDB() {
  try {
    // Connect to MongoDB without deprecated options
    await mongoose.connect(process.env.MONGO_URI);

    console.log('MongoDB connected...');

    const db = mongoose.connection.db;

    const collections = await db.listCollections({ name: 'contacts' }).toArray();

    if (collections.length === 0) {
      console.log("Collection 'contacts' does not exist. Creating it...");
      await db.createCollection('contacts');
      console.log("Collection 'contacts' created.");
    } else {
      console.log("Collection 'contacts' already exists.");
    }

  } catch (err) {
    console.error("MongoDB connection failed: ", err);
    process.exit(1);
  }
}


connectToDB();

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
