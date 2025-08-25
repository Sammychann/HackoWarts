// BluePulsePage.js
import React, { useState, useEffect } from "react";
import { DropletIcon, Menu } from "lucide-react";
import Sidebar from "./Sidebar"; // Import the Sidebar component

const Banner = ({ onMenuClick }) => (
  <div
    style={{
      background: "linear-gradient(135deg, #ffffff 0%, #0ea5e9 100%)",
      padding: "1rem",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
    }}
  >
    <Menu
      size={24}
      style={{ cursor: "pointer", marginLeft: "10px" }}
      onClick={onMenuClick} // Open sidebar
    />
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "10px",
        flex: 1,
        justifyContent: "flex-start",
      }}
    >
      <DropletIcon
        style={{
          color: "#0ea5e9",
          marginLeft: "50px",
        }}
      />
      <h1
        style={{
          fontSize: "2.5rem",
          fontWeight: "bold",
          margin: 0,
        }}
      >
        BluePulse
      </h1>
    </div>
    <div
      style={{
        color: "#1e40af",
        fontSize: "1.1rem",
        textAlign: "left",
        flex: 2,
      }}
    >
      Water body pollution prevention and monitoring
    </div>
  </div>
);

const NewsMarquee = () => {
  const news = [
    "MahaKumbh Mela: Millions gather for holy dip in the Ganges",
    "New water quality standards implemented in coastal regions",
    "Record rainfall improves reservoir levels nationwide",
    "International conference on water conservation next month",
  ];

  const [position, setPosition] = useState(100);

  useEffect(() => {
    const animation = setInterval(() => {
      setPosition((prev) => {
        if (prev <= -200) return 100;
        return prev - 0.05;
      });
    }, 16);

    return () => clearInterval(animation);
  }, []);

  return (
    <div
      style={{
        height: "20px",
        background: "#e0f2fe",
        padding: "0.5rem",
        overflow: "hidden",
        position: "relative",
        marginTop: "1px",
      }}
    >
      <div
        style={{
          whiteSpace: "nowrap",
          position: "absolute",
          left: `${position}%`,
          color: "#0369a1",
          fontWeight: "bold",
        }}
      >
        {news.join(" | ")}
      </div>
    </div>
  );
};

const InfoBox = ({ title, content }) => {
  return (
    <div
      style={{
        padding: "2rem",
        backgroundColor: "white",
        borderRadius: "8px",
        boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
        flex: 1,
        minWidth: "300px",
        textAlign: "center",
        transition: "all 0.3s ease-in-out",
        cursor: "pointer",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow =
          "0 8px 16px rgba(14, 165, 233, 0.6)";
        e.currentTarget.style.transform = "scale(1.05)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow =
          "0 4px 6px rgba(0, 0, 0, 0.1)";
        e.currentTarget.style.transform = "scale(1)";
      }}
    >
      <h2
        style={{
          color: "#0ea5e9",
          marginBottom: "1rem",
          fontSize: "1.5rem",
        }}
      >
        {title}
      </h2>
      <p
        style={{
          color: "#334155",
          lineHeight: "1.6",
        }}
      >
        {content}
      </p>
    </div>
  );
};

const DashboardFrame = () => {
  return (
    <div
      style={{
        width: "100%",
        height: "100vh",
        border: "none",
        marginTop: "2rem",
        display: "flex",
        justifyContent: "center",
      }}
    >
      <iframe
        src="http://localhost:8501" // Your Streamlit app URL
        width="100%"
        height="100%"
        style={{
          border: "2px solid #0E1117",
          borderRadius: "8px",
          boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
        }}
        title="Dashboard"
      ></iframe>
    </div>
  );
};

const BluePulsePage = () => {
  const [isSidebarOpen, setSidebarOpen] = useState(false);

  const toggleSidebar = () => setSidebarOpen(!isSidebarOpen);

  const infoBoxes = [
    {
      title: "Water Pollution Overview",
      content:
        "Water pollution occurs when harmful substances contaminate water bodies, degrading water quality and making it toxic to humans or the environment. Major sources include industrial waste, agricultural runoff, and urban sewage.",
    },
    {
      title: "Impact on Ecosystems",
      content:
        "Pollution in water bodies severely affects aquatic ecosystems, leading to loss of biodiversity, habitat destruction, and disruption of food chains. This can have far-reaching consequences for both marine life and human communities.",
    },
    {
      title: "Prevention Measures",
      content:
        "Preventing water pollution requires collective effort. Key strategies include proper waste disposal, reducing plastic use, implementing water treatment systems, and supporting environmental regulations that protect our water resources.",
    },
  ];

  return (
    <div style={{ backgroundColor: "#0E1117", minHeight: "100vh" }}>
      <Sidebar isOpen={isSidebarOpen} onClose={toggleSidebar} />
      <Banner onMenuClick={toggleSidebar} />
      <NewsMarquee />

      <div
        style={{
          maxWidth: "1400px",
          margin: "2rem auto",
          padding: "0 1rem",
          display: "flex",
          justifyContent: "space-between",
          gap: "1.5rem",
          flexWrap: "wrap",
        }}
      >
        {infoBoxes.map((box, index) => (
          <InfoBox key={index} title={box.title} content={box.content} />
        ))}
      </div>

      <DashboardFrame />
    </div>
  );
};

export default BluePulsePage;
