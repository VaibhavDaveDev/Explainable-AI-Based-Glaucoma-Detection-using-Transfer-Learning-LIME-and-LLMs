import { Form, Button } from "react-bootstrap";
import { useState } from "react";

/* eslint-disable react/prop-types */
export default function FormComponent({ setResult, setImages, setExplain, setIsLoading }) {
  const [formData, setFormData] = useState({
    ImgFile: null,
  });

  function handleInputChange(e) {
    const { name, type, files } = e.target;
    setFormData({
      ...formData,
      [name]: type === "file" ? files[0] : value,
    });
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setIsLoading(true);

    try {
      const formDataToSend = new FormData();
      formDataToSend.append("file", formData.ImgFile);  // Ensure the field name matches the FastAPI endpoint

      const predictions = await handleRequest(formDataToSend);
      setResult(predictions.predictions); // Update to set the predictions
      setImages(predictions.images); // Update to set images returned from FastAPI
      setExplain(predictions.groq_explanation); // Update to set explain returned from FastAPI
    } catch (error) {
      console.error("Error occurred while sending POST request:", error);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleRequest(formDataToSend) {
    const URL = import.meta.env.VITE_DEV_BACKEND_ENDPOINT; // Ensure this points to your FastAPI endpoint
    const options = {
      method: "POST",
      body: formDataToSend,
    };

    const res = await fetch(URL + "/predict/", options);
    if (res.status === 200) {
      const data = await res.json();
      return data; // Return the entire data object
    } else {
      throw new Error("Failed to fetch predictions");
    }
  }

  return (
    <Form id='form' onSubmit={handleSubmit}>
      <Form.Group controlId='ImgFile'>
        <Form.Label>Upload the respective scanned image file:</Form.Label>
        <Form.Control
          type='file'
          name='ImgFile'
          onChange={handleInputChange}
          required
        />
      </Form.Group>

      <Button variant='primary' type='submit'>
        Submit
      </Button>
    </Form>
  );
}
