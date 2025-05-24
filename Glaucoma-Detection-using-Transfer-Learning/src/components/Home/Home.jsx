import { Container, Row, Col, Button } from "react-bootstrap";
import { Link } from "react-router-dom";
import "./style.css";

export default function Home() {
  document.title = "Home | GlaucoDetect";
  return (
    <Container id='home'>
      <Row>
        <Col id='context'>
          <h1>Welcome to GlaucoDetect</h1>
          <p>
            Glaucoma is a slient killer of the world you see. It{"'"}s hard to
            spot and even harder to treat once it takes hold. You can treat it
            only at it{"'"}s early stages. How can you detect it when really
            hard to spot ?
          </p>
          <p>
            Preseting you <span>GlaucoDetect</span>, an AI-powered web
            application that helps you get your result that are accurate and
            fast.
          </p>
          <Link to='/predict'>
            <Button id="predict-button">
              Check your eye
            </Button>
          </Link>
        </Col>
        <Col id='img'></Col>
      </Row>
    </Container>
  );
}
