import { useState, useEffect, useRef } from 'react';
import { 
  Search, 
  AlertCircle, 
  Loader2, 
  Network,
  Sparkles,
  X,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  HelpCircle,
  Code,
  Eye,
  EyeOff,
  ZoomIn,
  Info
} from 'lucide-react';
import { Network as VisNetwork } from 'vis-network';
import { DataSet } from 'vis-data';

const API_BASE_URL = 'http://localhost:5001/api';

const NODE_COLORS = {
  Model: { 
    background: '#6366F1', 
    border: '#4F46E5', 
    highlight: { background: '#818CF8', border: '#6366F1' },
    hover: { background: '#818CF8', border: '#6366F1' }
  },
  Family: { 
    background: '#10B981', 
    border:  '#059669', 
    highlight: { background: '#34D399', border: '#10B981' },
    hover: { background: '#34D399', border: '#10B981' }
  },
  Centroid: { 
    background:  '#F59E0B', 
    border: '#D97706', 
    highlight:  { background: '#FBBF24', border: '#F59E0B' },
    hover: { background: '#FBBF24', border: '#F59E0B' }
  },
  FamilyCentroid: { 
    background:  '#F59E0B', 
    border: '#D97706', 
    highlight: { background: '#FBBF24', border:  '#F59E0B' },
    hover: { background: '#FBBF24', border:  '#F59E0B' }
  }
};

const DEFAULT_NODE_COLOR = { 
  background: '#6B7280', 
  border:  '#4B5563', 
  highlight:  { background: '#9CA3AF', border: '#6B7280' },
  hover: { background: '#9CA3AF', border: '#6B7280' }
};

const EDGE_COLORS = {
  IS_CHILD_OF: '#F87171',
  BELONGS_TO: '#60A5FA',
  HAS_CENTROID: '#FBBF24'
};

const DEFAULT_EDGE_COLOR = '#9CA3AF';

function FilterChip({ label, color, checked, onChange }) {
  return (
    <label className="flex items-center gap-3 cursor-pointer">
      <div 
        className={`relative w-10 h-6 rounded-full transition-all ${checked ? 'bg-gray-800' : 'bg-gray-300'}`}
        onClick={onChange}
      >
        <div 
          className={`absolute top-1 w-4 h-4 rounded-full bg-white shadow transition-all ${checked ? 'left-5' : 'left-1'}`}
        />
      </div>
      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
      <span className={`text-sm ${checked ? 'text-gray-900 font-medium' : 'text-gray-500'}`}>
        {label}
      </span>
    </label>
  );
}

export default function NLQueryPage() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [examples, setExamples] = useState([]);
  const [showExamples, setShowExamples] = useState(false);
  const [showCypher, setShowCypher] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [showFilters, setShowFilters] = useState(false);
  
  const [filters, setFilters] = useState({
    showModels: true,
    showFamilies: true,
    showCentroids: true,
    showIsChildOf: true,
    showBelongsTo: true,
    showHasCentroid: true
  });
  
  const networkRef = useRef(null);
  const containerRef = useRef(null);
  const nodesDataSetRef = useRef(null);
  const edgesDataSetRef = useRef(null);

  useEffect(() => {
    fetchExamples();
  }, []);

  const fetchExamples = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/nl-query/examples`);
      const data = await response.json();
      if (data.success) {
        setExamples(data.examples);
      }
    } catch (err) {
      console.error('Failed to fetch examples:', err);
    }
  };

  useEffect(() => {
    if (result?. data && result.data.node_count > 0 && containerRef.current) {
      const timer = setTimeout(() => {
        initializeNetwork(result.data);
      }, 150);
      return () => clearTimeout(timer);
    }
  }, [result]);

  useEffect(() => {
    updateNetworkFilters();
  }, [filters]);

  const isNodeVisible = (nodeType) => {
    if (nodeType === 'Model') return filters.showModels;
    if (nodeType === 'Family') return filters.showFamilies;
    if (nodeType === 'Centroid' || nodeType === 'FamilyCentroid') return filters.showCentroids;
    return true;
  };

  const isEdgeVisible = (edgeType) => {
    if (edgeType === 'IS_CHILD_OF') return filters.showIsChildOf;
    if (edgeType === 'BELONGS_TO') return filters.showBelongsTo;
    if (edgeType === 'HAS_CENTROID') return filters.showHasCentroid;
    return true;
  };

  const getNodeLabel = (node) => {
    const name = node.name || node.id || '';
    // Truncate long names
    if (name.length > 20) {
      return name.substring(0, 18) + '...';
    }
    return name;
  };

  const initializeNetwork = (graphData) => {
    if (!containerRef.current || !graphData) return;

    try {
      // For IS_CHILD_OF:  source is child, target is parent
      // We need to REVERSE the edges so parent points to child for correct hierarchy
      const hasTreeStructure = graphData.edges. some(e => e.type === 'IS_CHILD_OF');
      
      const nodes = graphData.nodes.map(node => ({
        id: node.id,
        label: getNodeLabel(node),
        title: `<div style="padding: 8px;font-family:system-ui;">
          <strong>${node.name || node.id}</strong><br/>
          <span style="color:#666;">Type: ${node.label}</span>
          ${node.properties?.is_foundation_model ? '<br/><span style="color:#4F46E5;">★ Foundation Model</span>' : ''}
        </div>`,
        shape: node.label === 'Family' ? 'diamond' : 
               (node.label === 'Centroid' || node.label === 'FamilyCentroid') ? 'star' : 'dot',
        color: NODE_COLORS[node.label] || DEFAULT_NODE_COLOR,
        font: { 
          color: '#1F2937',
          size:  12,
          face: 'system-ui, sans-serif',
          strokeWidth: 2,
          strokeColor: '#ffffff'
        },
        size: node.label === 'Family' ? 30 : 
              (node.label === 'Centroid' || node.label === 'FamilyCentroid') ? 25 : 20,
        nodeType: node.label,
        originalData: node,
        hidden: !isNodeVisible(node. label),
        borderWidth: 2,
        shadow: {
          enabled: true,
          color: 'rgba(0,0,0,0.1)',
          size: 5,
          x: 0,
          y: 2
        }
      }));

      // Process edges - reverse IS_CHILD_OF for correct tree direction
      const edges = graphData. edges.map((edge, index) => {
        const isChildOf = edge.type === 'IS_CHILD_OF';
        
        return {
          id: edge.id || `edge-${index}`,
          // REVERSE direction for IS_CHILD_OF so tree flows top-down (parent -> child)
          from: isChildOf ? edge.target : edge.source,
          to: isChildOf ? edge.source : edge.target,
          // Don't show label on edges - too cluttered
          // label: edge.type,
          arrows: { 
            to: { 
              enabled: true, 
              type: 'arrow', 
              scaleFactor: 0.6
            } 
          },
          color: {
            color: EDGE_COLORS[edge.type] || DEFAULT_EDGE_COLOR,
            highlight: EDGE_COLORS[edge.type] || DEFAULT_EDGE_COLOR,
            hover: EDGE_COLORS[edge.type] || DEFAULT_EDGE_COLOR,
            opacity: 0.8
          },
          width: 1.5,
          edgeType: edge.type,
          hidden: !isEdgeVisible(edge.type),
          smooth: {
            enabled: true,
            type: 'cubicBezier',
            forceDirection: 'vertical',
            roundness: 0.5
          },
          hoverWidth: 2
        };
      });

      nodesDataSetRef.current = new DataSet(nodes);
      edgesDataSetRef.current = new DataSet(edges);

      const options = {
        nodes: {
          borderWidth: 2,
          shadow: true
        },
        edges: {
          width: 1.5,
          smooth: {
            enabled: true,
            type: 'cubicBezier',
            forceDirection:  'vertical',
            roundness: 0.5
          },
          selectionWidth: 2
        },
        layout: hasTreeStructure ? {
          hierarchical: {
            enabled: true,
            direction: 'UD',              // Up to Down (root at TOP)
            sortMethod: 'directed',       // Use edge direction
            levelSeparation: 100,         // Vertical spacing between levels
            nodeSpacing: 80,              // Horizontal spacing between nodes
            treeSpacing: 120,             // Space between separate trees
            blockShifting: true,
            edgeMinimization: true,
            parentCentralization: true
          }
        } : {
          improvedLayout: true,
          randomSeed: 42
        },
        physics: {
          enabled: ! hasTreeStructure,     // Disable physics for tree layout
          solver: 'forceAtlas2Based',
          forceAtlas2Based: {
            gravitationalConstant: -50,
            centralGravity:  0.01,
            springLength: 100,
            springConstant: 0.08
          },
          stabilization: {
            enabled: true,
            iterations: 150
          }
        },
        interaction: {
          hover: true,
          tooltipDelay: 100,
          zoomView: true,
          dragView: true,
          dragNodes: true,              // Allow dragging nodes
          multiselect: true,
          navigationButtons: false,
          keyboard: {
            enabled: true,
            bindToWindow: false
          },
          zoomSpeed: 0.5
        }
      };

      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }

      networkRef.current = new VisNetwork(
        containerRef.current,
        { nodes:  nodesDataSetRef.current, edges: edgesDataSetRef.current },
        options
      );

      networkRef.current.on('click', (params) => {
        if (params.nodes. length > 0) {
          const nodeId = params.nodes[0];
          const node = nodesDataSetRef. current.get(nodeId);
          setSelectedNode(node?. originalData || null);
        } else {
          setSelectedNode(null);
        }
      });

      networkRef.current.on('doubleClick', (params) => {
        if (params.nodes. length > 0) {
          networkRef.current.focus(params.nodes[0], {
            scale: 1.2,
            animation: { duration: 400, easingFunction: 'easeOutQuad' }
          });
        }
      });

      // After layout is done, enable physics briefly for fine-tuning then disable
      if (hasTreeStructure) {
        networkRef.current.once('stabilized', () => {
          networkRef.current.setOptions({ physics: { enabled: false } });
          networkRef.current.fit({ 
            animation: { duration: 400, easingFunction: 'easeOutQuad' } 
          });
        });
      } else {
        setTimeout(() => {
          if (networkRef.current) {
            networkRef.current.fit({ animation: { duration: 400 } });
          }
        }, 500);
      }

    } catch (err) {
      console.error('Failed to initialize network:', err);
    }
  };

  const updateNetworkFilters = () => {
    if (!nodesDataSetRef.current || ! edgesDataSetRef.current) return;

    try {
      const nodeUpdates = [];
      nodesDataSetRef.current.forEach(node => {
        nodeUpdates.push({ id: node.id, hidden: !isNodeVisible(node. nodeType) });
      });
      if (nodeUpdates.length > 0) {
        nodesDataSetRef.current.update(nodeUpdates);
      }

      const edgeUpdates = [];
      edgesDataSetRef.current.forEach(edge => {
        edgeUpdates.push({ id: edge.id, hidden: !isEdgeVisible(edge.edgeType) });
      });
      if (edgeUpdates.length > 0) {
        edgesDataSetRef.current.update(edgeUpdates);
      }
    } catch (err) {
      console.error('Failed to update filters:', err);
    }
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);
    setSelectedNode(null);

    try {
      const response = await fetch(`${API_BASE_URL}/nl-query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question:  question.trim() })
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
        if (data.message) {
          setError({ type: 'info', message: data.message });
        }
      } else {
        setError({ type: 'error', message: data.error || 'Query failed' });
        setResult(null);
      }
    } catch (err) {
      console.error('Query failed:', err);
      setError({ type: 'error', message:  'Connection failed.  Please check that the backend is running.' });
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (example) => {
    setQuestion(example.question);
    setShowExamples(false);
  };

  const handleReset = () => {
    setQuestion('');
    setResult(null);
    setError(null);
    setSelectedNode(null);
    if (networkRef.current) {
      networkRef.current.destroy();
      networkRef.current = null;
    }
    nodesDataSetRef.current = null;
    edgesDataSetRef.current = null;
  };

  const handleFitView = () => {
    if (networkRef.current) {
      networkRef.current.fit({ animation: { duration: 400 } });
    }
  };

  const toggleFilter = (filterName) => {
    setFilters(prev => ({ ...prev, [filterName]: !prev[filterName] }));
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Explore Graph with Natural Language
        </h1>
        <p className="text-gray-600">
          Ask questions in English or Italian to explore relationships between models, families, and centroids. 
        </p>
      </div>

      {/* Search Bar */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex items-center gap-2 mb-2">
            <Sparkles className="h-5 w-5 text-indigo-600" />
            <h2 className="text-lg font-semibold text-gray-900">Ask the Graph</h2>
          </div>
          
          <div className="relative">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g., Show all foundation models..."
              className="w-full px-4 py-3 pr-28 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-gray-900 placeholder-gray-400"
              maxLength={500}
              disabled={loading}
            />
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
              {question && (
                <button
                  type="button"
                  onClick={() => setQuestion('')}
                  className="p-1. 5 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
              <button
                type="submit"
                disabled={! question.trim() || loading}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 font-medium"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                {loading ? 'Searching...' :  'Search'}
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between text-sm">
            <button
              type="button"
              onClick={() => setShowExamples(!showExamples)}
              className="flex items-center gap-1.5 text-indigo-600 hover:text-indigo-800 font-medium"
            >
              <HelpCircle className="h-4 w-4" />
              Example queries
              {showExamples ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>
            <span className="text-gray-400">{question.length}/500</span>
          </div>

          {showExamples && (
            <div className="grid grid-cols-1 md: grid-cols-2 lg:grid-cols-3 gap-2 mt-3 p-4 bg-gray-50 rounded-xl border border-gray-100">
              {examples.map((example, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => handleExampleClick(example)}
                  className="text-left p-3 hover:bg-white rounded-lg transition-all hover:shadow-sm border border-transparent hover:border-gray-200"
                >
                  <div className="font-medium text-gray-900">{example.question}</div>
                  <div className="text-gray-500 text-xs mt-1">{example.description}</div>
                </button>
              ))}
            </div>
          )}
        </form>
      </div>

      {/* Graph Visualization */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold flex items-center gap-2 text-gray-900">
            <Network className="h-5 w-5 text-indigo-600" />
            Graph Visualization
          </h2>
          
          <div className="flex items-center gap-3">
            {result?. data && (
              <span className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                {result.data.node_count} nodes, {result.data.edge_count} edges
              </span>
            )}
            {result?.data && result.data.node_count > 0 && (
              <button
                onClick={handleFitView}
                className="p-2 text-gray-600 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg"
                title="Fit to view"
              >
                <ZoomIn className="h-4 w-4" />
              </button>
            )}
            <button
              onClick={handleReset}
              className="p-2 text-gray-600 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg"
              title="Reset"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className={`mb-4 p-4 rounded-lg flex items-start gap-3 ${
            error.type === 'info' ? 'bg-blue-50 border border-blue-100' : 'bg-red-50 border border-red-100'
          }`}>
            {error.type === 'info' ?  (
              <Info className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
            ) : (
              <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
            )}
            <div className={error.type === 'info' ?  'text-blue-800' : 'text-red-800'}>
              {error.message}
            </div>
          </div>
        )}

        {/* Cypher Query Display */}
        {result?.cypher && (
          <div className="mb-4">
            <button
              onClick={() => setShowCypher(!showCypher)}
              className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 font-medium"
            >
              <Code className="h-4 w-4" />
              {showCypher ? 'Hide' : 'Show'} Cypher Query
              {showCypher ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>
            
            {showCypher && (
              <pre className="mt-3 p-4 bg-gray-100 text-gray-800 rounded-xl text-sm overflow-x-auto font-mono border border-gray-200">
                <code>{result.cypher}</code>
              </pre>
            )}
          </div>
        )}

        {/* Filters */}
        {result?.data && result.data.node_count > 0 && (
          <div className="mb-4">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 font-medium"
            >
              {showFilters ? <EyeOff className="h-4 w-4" /> :  <Eye className="h-4 w-4" />}
              Visualization Filters
              {showFilters ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>

            {showFilters && (
              <div className="mt-3 p-5 bg-gray-50 rounded-xl border border-gray-100">
                <div className="grid grid-cols-1 md: grid-cols-2 gap-6">
                  <div>
                    <div className="text-sm font-semibold text-gray-700 mb-3 uppercase tracking-wide">Nodes</div>
                    <div className="space-y-3">
                      <FilterChip label="Model" color="#6366F1" checked={filters.showModels} onChange={() => toggleFilter('showModels')} />
                      <FilterChip label="Family" color="#10B981" checked={filters.showFamilies} onChange={() => toggleFilter('showFamilies')} />
                      <FilterChip label="Centroid" color="#F59E0B" checked={filters.showCentroids} onChange={() => toggleFilter('showCentroids')} />
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-gray-700 mb-3 uppercase tracking-wide">Relationships</div>
                    <div className="space-y-3">
                      <FilterChip label="IS_CHILD_OF" color="#F87171" checked={filters.showIsChildOf} onChange={() => toggleFilter('showIsChildOf')} />
                      <FilterChip label="BELONGS_TO" color="#60A5FA" checked={filters. showBelongsTo} onChange={() => toggleFilter('showBelongsTo')} />
                      <FilterChip label="HAS_CENTROID" color="#FBBF24" checked={filters.showHasCentroid} onChange={() => toggleFilter('showHasCentroid')} />
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Graph Container */}
        <div 
          ref={containerRef}
          className="w-full h-[600px] border border-gray-200 rounded-xl bg-white overflow-hidden"
          style={{ minHeight: '600px' }}
        >
          {! result && !loading && (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <div className="w-20 h-20 rounded-full bg-indigo-100 flex items-center justify-center mb-4">
                <Network className="h-10 w-10 text-indigo-400" />
              </div>
              <p className="text-lg font-medium text-gray-700">Ask a question to visualize the graph</p>
              <p className="text-sm mt-1 text-gray-500">Use natural language to explore data</p>
            </div>
          )}
          
          {loading && (
            <div className="flex flex-col items-center justify-center h-full">
              <Loader2 className="h-12 w-12 animate-spin text-indigo-600 mb-4" />
              <p className="text-gray-600 font-medium">Processing query...</p>
            </div>
          )}
        </div>

        {/* Selected Node Details */}
        {selectedNode && (
          <div className="mt-4 p-5 bg-indigo-50 rounded-xl border border-indigo-100">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-indigo-900 flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor:  NODE_COLORS[selectedNode. label]?. background || '#6B7280' }}
                />
                Node Details
              </h3>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-indigo-400 hover:text-indigo-600 p-1 hover:bg-indigo-100 rounded"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-white/60 rounded-lg p-3">
                <span className="text-indigo-500 text-xs font-medium uppercase">Type</span>
                <div className="font-semibold text-gray-900 mt-1">{selectedNode. label}</div>
              </div>
              <div className="bg-white/60 rounded-lg p-3">
                <span className="text-indigo-500 text-xs font-medium uppercase">ID</span>
                <div className="font-mono text-gray-900 mt-1 text-xs truncate">{selectedNode.id}</div>
              </div>
              {selectedNode.name && (
                <div className="bg-white/60 rounded-lg p-3">
                  <span className="text-indigo-500 text-xs font-medium uppercase">Name</span>
                  <div className="font-semibold text-gray-900 mt-1 truncate">{selectedNode.name}</div>
                </div>
              )}
              {selectedNode. properties?.is_foundation_model && (
                <div className="bg-white/60 rounded-lg p-3 flex items-center">
                  <span className="px-2 py-1 bg-indigo-600 text-white rounded-full text-xs font-medium">
                    Foundation Model
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Legend */}
        {result?.data && result.data.node_count > 0 && (
          <div className="mt-4 flex flex-wrap items-center gap-4 text-xs text-gray-600 p-3 bg-gray-50 rounded-lg">
            <span className="font-semibold text-gray-700">Legend:</span>
            <div className="flex items-center gap-1. 5">
              <div className="w-3 h-3 rounded-full bg-[#6366F1]" />
              <span>Model</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rotate-45 bg-[#10B981]" />
              <span>Family</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 bg-[#F59E0B]" style={{ clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)' }} />
              <span>Centroid</span>
            </div>
            <span className="text-gray-300">|</span>
            <div className="flex items-center gap-1.5">
              <div className="w-4 h-0.5 bg-[#F87171]" />
              <span>IS_CHILD_OF</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-4 h-0.5 bg-[#60A5FA]" />
              <span>BELONGS_TO</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-4 h-0.5 bg-[#FBBF24]" />
              <span>HAS_CENTROID</span>
            </div>
            <span className="text-gray-300">|</span>
            <span className="text-gray-500 italic">Drag nodes to reposition • Double-click to zoom</span>
          </div>
        )}
      </div>
    </div>
  );
}