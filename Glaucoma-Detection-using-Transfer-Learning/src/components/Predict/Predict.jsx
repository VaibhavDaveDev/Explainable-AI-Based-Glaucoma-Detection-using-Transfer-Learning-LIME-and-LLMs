import { Container, Row, Col } from "react-bootstrap";
import { useState, useEffect } from "react";
import FormComponent from "./FormComponent";
import "./style.css";
import safe from "../../assets/img/safe.png";
import warning from "../../assets/img/warning.png";
import danger from "../../assets/img/danger.png";
import detection from "../../assets/img/detection.png";
import loading from "../../assets/img/loading.png";

// eslint-disable-next-line react/prop-types
function RenderResult({ result, images, isLoading }) {
  if (!result || isNaN(result)) {
    if (isLoading) return <img id="loadingimg" className="loader" src={loading} alt="loading" />;
    return (
      <img id="detectionimg" src={detection} alt="detection image" style={{ width: "100%" }} />
    );
  }

  let img, description;
  const floatResult = parseFloat(result);

  if (floatResult <= 0.4) img = { name: "Safe", src: safe };
  else if (floatResult <= 0.7) img = { name: "Warning", src: warning };
  else img = { name: "Danger", src: danger };

  switch (img.name) {
    case "Safe":
      description = `Your eye is in good condition. You are safe. Keep it up.`;
      break;
    case "Warning":
      description = `Your eye is in warning condition. Please consult with a doctor as soon as possible.`;
      break;
    case "Danger":
      description = `Your eye is in danger condition. Please consult with a doctor immediately.`;
      break;
    default:
      description = `Something went wrong. Please try again.`;
  }

  return (
    <>
      <img id="statusimg" src={img.src} alt={`${img.name} image`} />
      <h2>
        Status: <b>{img.name}</b>
      </h2>
      <h2>
        Probability of Glaucoma: <b>{Math.round(floatResult * 1000) / 10}%</b>
      </h2>
      <p>{description}</p>
    </>
  );
}

function AnalyzedImages({ images, llmExplaination }) {
  const formatKey = (key) => {
    return key
      .replace(/_/g, " ") // Replace underscores with spaces
      .replace(/\b\w/g, (char) => char.toUpperCase()); // Capitalize the first letter of each word
  };
  // Render JSON dynamically
  const renderJsonData = (data) => {
    return Object.keys(data).map((key) => {
      const value = data[key];
      const formattedKey = formatKey(key);
      if (Array.isArray(value)) {
        // Handle arrays
        if (value.length > 0 && typeof value[0] === "object") {
          // If array contains objects, render them
          return (
            <div key={key} className="json-section">
              {/* <h1>{formattedKey}</h1> */}
            </div>
          );
        } else {
          // If array contains primitives, render them
          return (
            <div key={key} className="json-section">
              <h1>{formattedKey}</h1>
              <ul>
                {value.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          );
        }
      } else if (typeof value === "object") {
        // Handle nested objects
        return (
          <div key={key} className="json-section">

            {renderJsonData(value)}
          </div>
        );
      } else {
        // Handle primitive values
        return (
          <div key={key} className="json-section">
            {/* { <h1>{formattedKey}</h1> } */}
            <strong>→ {value}</strong>
            {"\n"}
          </div>
        );
      }
    });
  };

  return (
    images && (
      <div>
        <h3>LIME Explanations:</h3>
        <Row>
          <Col>
            <h4 id="Original Image">Original Image</h4>
            <p>Displays the original image being analyzed.
            </p>
            <img className="AnalyzedImage" src={images.original_image} alt="Original" />
          </Col>

          <Col>
            <h4 id="Superpixels Image">Superpixels Image</h4>
            <p>Visualizes the contributions of superpixels</p>
            <img className="AnalyzedImage" src={images.superpixels_image} alt="Superpixels" />
          </Col>
        </Row>
        <Row>
          <Col>
            <h4 id="LIME Explanation">LIME Positive and Negative Contributions</h4>
            <p>Visualizes the contributions of different superpixels, highlighting both positive and negative impacts.
            </p>
            <img className="AnalyzedImage" src={images.lime_explanation} alt="LIME Explanation" />
          </Col>

          <Col>
            <h4 id="LIME Positive Contributions">LIME Positive Contributions</h4>
            <p>Shows areas that positively contribute to the prediction, masking out the rest
            </p>
            <img className="AnalyzedImage" src={images.lime_positive} alt="LIME Positive Contributions" />
          </Col>
        </Row>
        <Row>
          <Col>
            <h4 id="Top Contributing Superpixels">Top Contributing Superpixels</h4>
            <p>Highlights the top contributing superpixels over the original image to show their importance.
            </p>
            <img className="AnalyzedImage" src={images.top_contributing} alt="Top Contributing Superpixels" />
          </Col>
          <Col>
            <h4 id="Mask Overlay">LIME Mask Overlay on Original Image</h4>
            <p>Overlays the LIME mask on the original image to visualize the areas influencing the prediction.
            </p>
            <img className="AnalyzedImage" src={images.mask_overlay} alt="Mask Overlay" />
          </Col>

        </Row>

        <h4 id="Perturbed Images">Perturbed Images</h4>
        <p>Displays perturbed images with certain superpixels turned off to understand their contribution.
        </p>
        <Row>
          <Col>
            <img className="AnalyzedImage" src={images.perturbed_images[0]} alt="Perturbed Image 1" />
          </Col>
          <Col>
            {images.perturbed_images[1] && (
              <img className="AnalyzedImage" src={images.perturbed_images[1]} alt="Perturbed Image 2" />
            )}
          </Col>
        </Row>
        <Row>
          <Col>
            {images.perturbed_images[2] && (
              <img className="AnalyzedImage" src={images.perturbed_images[2]} alt="Perturbed Image 3" />
            )}
          </Col>
          <Col>
            {images.perturbed_images[3] && (
              <img className="AnalyzedImage" src={images.perturbed_images[3]} alt="Perturbed Image 4" />
            )}
          </Col>
        </Row>



        {/* Render GROQ Explanation */}
        {/* <Row>
          <Col>
            <div class="image-with-heading">
              <img id="magic" src="src/assets/img/Google_Bard_logo.svg.png" alt="Description of image"></img>
              <h1 id="GROQ Explanation">AI GENERATED</h1>
            </div>
          </Col>
        </Row> */}

        {/* Superpixel Importance Visualization */}
        {/* <Row>
          <Col>
          <h1></h1>
            <h1 id="Superpixel Importance">Superpixel Importance Visualization</h1>
            <h4>Annotated image showing importance scores of superpixels, highlighting contributions.</h4>
            <img className="AnalyzedImage" id="gen" src={images.superpixel_importance} alt="Superpixel Importance" />
          </Col>
        </Row> */}

        {/* <Row>
          <Col>
          <h1></h1>
            <h3> Explanation :-</h3>
            <div className="GroqExplanation">{renderJsonData(llmExplaination)}</div>
          </Col>
        </Row> */}

        <Row>
          <Col>
            <div className="groq-container">
              {/* Header with Logo */}
              <div className="image-with-heading">
                <img id="magic" src="src/assets/img/Google_Bard_logo.svg.png" alt="Description of image" />
                <h1 id="GROQ-Explanation">AI GENERATED</h1>
              </div>

              {/* Superpixel Importance Visualization */}
              <div className="visualization-section">
                <h1 id="Superpixel-Importance">Superpixel Importance Visualization</h1>
                <img className="AnalyzedImage" id="gen" src={images.superpixel_importance} alt="Superpixel Importance" />
                <h4 id= "Superpixel-Importance-description" >Annotated image showing importance scores of superpixels, highlighting contributions.</h4>
              </div>

              {/* Explanation */}
              <div className="explanation-section">
                <h3>Explanation:</h3>
                <div className="GroqExplanation">{renderJsonData(llmExplaination)}</div>
              </div>
            </div>
          </Col>
        </Row>



      </div>
    )
  );
}

export default function Predict() {
  const [result, setResult] = useState();
  const [images, setImages] = useState(); // State to hold images from predictions
  const [llmExplaination, setExplain] = useState();
  const [isLoading, setIsLoading] = useState(false);
  const hasAnalyzedImages = images && Object.keys(images).length > 0; // Check if images are present

  // Update body class based on analyzed images presence
  useEffect(() => {
    if (hasAnalyzedImages) {
      document.body.classList.add('expanded');
    } else {
      document.body.classList.remove('expanded');
    }
  }, [hasAnalyzedImages]);

  return (
    <div className={`beta ${hasAnalyzedImages ? 'expanded' : ''}`}>
      <Container id='main'>
        <Row>
          <Col>
            <h1>Check your Eye</h1>
            <p>
              To analyze the condition of the eye, the ML model mainly requires
              your retinal scan.
            </p>
            <FormComponent {...{ setResult, setImages, setExplain, setIsLoading }} />
          </Col>
          <Col id='display-results'>
            <RenderResult result={result} images={images} isLoading={isLoading} />
          </Col>
        </Row>

        {/* New Row for Analyzed Images */}
        <Row>
          <Col>
            <AnalyzedImages images={images} llmExplaination={llmExplaination} />

          </Col>
        </Row>
      </Container>
    </div>
  );
}
