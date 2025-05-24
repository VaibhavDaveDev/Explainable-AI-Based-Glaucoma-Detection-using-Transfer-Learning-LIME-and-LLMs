import mongoose from 'mongoose';

const contactSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  email: {
    type: String,
    required: true,
  },
  message: {
    type: String,
    required: true,
  },
  date: {
    type: String, // To store the date in a formatted string
    required: true,
  },
  time: {
    type: String, // To store the time in a formatted string (12-hour format)
    required: true,
  },
});

const Contact = mongoose.model('Contact', contactSchema);
export default Contact;
