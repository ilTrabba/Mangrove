import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import ModelsPage from './pages/ModelsPage';
import ModelDetailPage from './pages/ModelDetailPage';
import AddModelPage from './pages/AddModelPage';
import GraphPage from './pages/GraphPage';
import NLQueryPage from './pages/NLQueryPage';
import './App.css';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <main>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/models/:id" element={<ModelDetailPage />} />
            <Route path="/add-model" element={<AddModelPage />} />
            <Route path="/graph" element={<GraphPage />} />
            <Route path="/explore" element={<NLQueryPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;