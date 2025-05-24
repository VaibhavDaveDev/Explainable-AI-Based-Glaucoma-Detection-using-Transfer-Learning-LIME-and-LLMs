import express from 'express';
import Contact from '../models/Contact.js';  // Adjust file path

const router = express.Router();

// POST route to handle form submission
router.post('/', async (req, res) => {
  const { name, email, message } = req.body;

  // Validate input fields
  if (!name || !email || !message) {
    return res.status(400).json({ msg: 'Please enter all fields' });
  }

  try {
    // Get current date and time in IST
    const now = new Date();
    
    // Format the current date in 'dd-mm-yyyy' format
    const date = now.toLocaleString('en-IN', {
      timeZone: 'Asia/Kolkata', // Indian Standard Time
      year: 'numeric',
      month: 'numeric',
      day: 'numeric',
    });

    // Format the current time in 12-hour format
    const time = now.toLocaleString('en-IN', {
      timeZone: 'Asia/Kolkata', // Indian Standard Time
      hour: 'numeric',
      minute: 'numeric',
      hour12: true,
    });

    // Create a new contact object
    const newContact = new Contact({
      name,
      email,
      message,
      date, // Store the formatted date
      time, // Store the formatted time
    });

    // Save the contact to the database
    const savedContact = await newContact.save();

    // Respond with success message and saved contact
    res.status(200).json({
      success: true,
      msg: 'Contact saved successfully',
      contact: savedContact,
    });
  } catch (error) {
    console.error('Error saving contact:', error); // Log the error
    res.status(500).json({ success: false, msg: 'Server error' });
  }
});

export default router;
