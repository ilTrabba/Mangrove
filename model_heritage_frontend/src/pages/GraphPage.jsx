import { useState, useEffect } from 'react';
import { AlertCircle, Loader2, RefreshCw, Database, Network } from 'lucide-react';

const API_BASE_URL = 'http://localhost:5001/api';

export default function GraphPage() {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [error, setError] = useState(null);

  const fetchGraphStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/graph/status`);
      const data = await response.json();
      setStatus(data);
    } catch (err) {
      console.error('Failed to fetch graph status:', err);
      setError('Failed to fetch graph status');
    }
  };

  const fetchGraphData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/graph/full`);
      const data = await response.json();
      
      if (response.ok) {
        setGraphData(data);
      } else {
        setError(data.error || 'Failed to fetch graph data');
        setGraphData({ nodes: [], edges: [] });
      }
    } catch (err) {
      console.error('Failed to fetch graph data:', err);
      setError('Failed to fetch graph data');
      setGraphData({ nodes: [], edges: [] });
    } finally {
      setLoading(false);
    }
  };

  const syncGraphData = async () => {
    setSyncing(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/graph/sync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ clear_existing: false })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        alert(`Sync successful: ${data.message}`);
        fetchGraphData();
        fetchGraphStatus();
      } else {
        setError(data.error || 'Sync failed');
      }
    } catch (err) {
      console.error('Failed to sync graph data:', err);
      setError('Failed to sync graph data');
    } finally {
      setSyncing(false);
    }
  };

  useEffect(() => {
    fetchGraphStatus();
    fetchGraphData();
  }, []);

  const renderStatusCard = () => (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold flex items-center">
          <Database className="h-5 w-5 mr-2" />
          Graph Database Status
        </h2>
        <button
          onClick={fetchGraphStatus}
          className="text-blue-600 hover:text-blue-800"
        >
          <RefreshCw className="h-4 w-4" />
        </button>
      </div>
      
      {status && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <div className="font-medium text-gray-600">Neo4j Status</div>
            <div className={status.neo4j_connected ? 'text-green-600' : 'text-red-600'}>
              {status.neo4j_connected ? '✓ Connected' : '✗ Disconnected'}
            </div>
          </div>
          <div>
            <div className="font-medium text-gray-600">SQLite Models</div>
            <div className="text-gray-900">{status.sqlite_models || 0}</div>
          </div>
          <div>
            <div className="font-medium text-gray-600">Graph Nodes</div>
            <div className="text-gray-900">{status.neo4j_nodes || 0}</div>
          </div>
          <div>
            <div className="font-medium text-gray-600">Graph Edges</div>
            <div className="text-gray-900">{status.neo4j_edges || 0}</div>
          </div>
        </div>
      )}
    </div>
  );

  const renderControls = () => (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <h2 className="text-xl font-semibold mb-4">Graph Controls</h2>
      <div className="flex flex-wrap gap-4">
        <button
          onClick={fetchGraphData}
          disabled={loading}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <RefreshCw className="h-4 w-4 mr-2" />}
          Refresh Graph
        </button>
        
        <button
          onClick={syncGraphData}
          disabled={syncing || !status?.neo4j_connected}
          className="flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
        >
          {syncing ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Database className="h-4 w-4 mr-2" />}
          Sync Data
        </button>
      </div>
      
      {!status?.neo4j_connected && (
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
          <div className="flex items-center">
            <AlertCircle className="h-4 w-4 text-yellow-600 mr-2" />
            <span className="text-yellow-800 text-sm">
              Neo4j is not connected. Graph features are unavailable.
            </span>
          </div>
        </div>
      )}
    </div>
  );

  const renderGraphVisualization = () => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold mb-4 flex items-center">
        <Network className="h-5 w-5 mr-2" />
        Graph Visualization
      </h2>
      
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <div className="flex items-center">
            <AlertCircle className="h-4 w-4 text-red-600 mr-2" />
            <span className="text-red-800 text-sm">{error}</span>
          </div>
        </div>
      )}
      
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
          <span className="ml-2 text-gray-600">Loading graph data...</span>
        </div>
      ) : graphData.nodes.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <Network className="h-12 w-12 mx-auto mb-4 text-gray-300" />
          <p>No graph data available</p>
          <p className="text-sm mt-2">Try syncing data from SQLite to Neo4j</p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Graph Statistics */}
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="font-medium text-gray-600">Total Nodes</div>
                <div className="text-lg font-semibold">{graphData.node_count || graphData.nodes.length}</div>
              </div>
              <div>
                <div className="font-medium text-gray-600">Total Edges</div>
                <div className="text-lg font-semibold">{graphData.edge_count || graphData.edges.length}</div>
              </div>
              <div>
                <div className="font-medium text-gray-600">Model Nodes</div>
                <div className="text-lg font-semibold">
                  {graphData.nodes.filter(n => n.label === 'Model').length}
                </div>
              </div>
              <div>
                <div className="font-medium text-gray-600">Family Nodes</div>
                <div className="text-lg font-semibold">
                  {graphData.nodes.filter(n => n.label === 'Family').length}
                </div>
              </div>
            </div>
          </div>
          
          {/* Simple node list for now */}
          <div className="space-y-2">
            <h3 className="font-medium text-gray-700">Nodes in Graph:</h3>
            <div className="max-h-64 overflow-y-auto space-y-1">
              {graphData.nodes.map((node, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-2 bg-gray-50 rounded text-sm"
                >
                  <div className="flex items-center">
                    <div
                      className="w-3 h-3 rounded-full mr-2"
                      style={{ backgroundColor: node.color || '#808080' }}
                    ></div>
                    <span className="font-medium">{node.label}</span>
                    <span className="ml-2 text-gray-600">{node.name || node.id}</span>
                  </div>
                  <span className="text-xs text-gray-500">{node.id}</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="text-xs text-gray-500 mt-4">
            Note: This is a basic representation. In a full implementation, this would be replaced 
            with an interactive graph visualization library like D3.js or vis.js.
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">MANGROVE Graph</h1>
        <p className="text-gray-600">
          Visualize model relationships and family structures using Neo4j graph database.
        </p>
      </div>

      {renderStatusCard()}
      {renderControls()}
      {renderGraphVisualization()}
    </div>
  );
}