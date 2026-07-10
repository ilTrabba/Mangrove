import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { 
  ArrowLeft, 
  FileText, 
  Users, 
  Clock, 
  Hash, 
  Database,
  TrendingUp,
  GitBranch,
  Info,
  Scale,
  Tag,
  Link as LinkIcon,
  Sparkles,
  ExternalLink,
  Download
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function ModelDetailPage() {
  const { id } = useParams();
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [readmeContent, setReadmeContent] = useState(null);
  const [readmeLoading, setReadmeLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    fetchModel();
  }, [id]);

  useEffect(() => {
    if (model?.readme_uri) {
      fetchReadme();
    }
  }, [model?.readme_uri]);

  const fetchModel = async () => {
    try {
      const response = await fetch(`http://localhost:5002/api/models/${id}`);
      if (!response.ok) throw new Error('Model not found');
      const data = await response.json();
      setModel(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchReadme = async () => {
    if (!model?.readme_uri) return;
    
    setReadmeLoading(true);
    try {
      const response = await fetch(`http://localhost:5002/api/models/${id}/readme`);
      if (response.ok) {
        const data = await response.json();
        setReadmeContent(data.content);
      }
    } catch (err) {
      console.error('Failed to fetch README:', err);
    } finally {
      setReadmeLoading(false);
    }
  };

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const response = await fetch(`http://localhost:5002/api/models/${id}/download`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Download failed');
      }
      
      // Get filename from Content-Disposition header or use default
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `${model?. name || id}.safetensors`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition. match(/filename="? (. +)"?/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }
      
      // Create blob and download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
    } catch (err) {
      console.error('Download failed:', err);
      alert(`Download failed: ${err.message}`);
    } finally {
      setDownloading(false);
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000000) return (num / 1000000000).toFixed(1) + 'B';
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num?.toString() || '0';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'ok': return 'bg-green-100 text-green-800';
      case 'processing': return 'bg-yellow-100 text-yellow-800';
      case 'error': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getDatasetVerificationBadge = (verified) => {
    if (verified === null || verified === undefined) {
      return { text: '🟡 Verifying', className: 'bg-yellow-100 text-yellow-800' };
    }
    if (verified === true) {
      return { text: '🟢 Verified', className: 'bg-green-100 text-green-800' };
    }
    return { text: '🔴 Unverified', className: 'bg-red-100 text-red-800' };
  };

  const getLicenseColor = (license) => {
    const colors = {
      'MIT': 'bg-green-100 text-green-800',
      'Apache-2.0': 'bg-orange-100 text-orange-800',
      'GPL-3.0': 'bg-blue-100 text-blue-800',
      'BSD-3-Clause': 'bg-purple-100 text-purple-800',
      'CC-BY-NC-4.0': 'bg-pink-100 text-pink-800',
      'Proprietary': 'bg-red-100 text-red-800'
    };
    return colors[license] || 'bg-gray-100 text-gray-800';
  };

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center">
          <p className="text-red-600 mb-4">Error: {error}</p>
          <Link to="/models">
            <Button>Back to Models</Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <Link to="/models">
          <Button variant="ghost" className="mb-4">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Models
          </Button>
        </Link>
        
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <h1 className="text-3xl font-bold text-gray-900">{model.name}</h1>
              {model.is_foundation_model && (
                <Badge className="bg-purple-100 text-purple-800">
                  <Sparkles className="h-3 w-3 mr-1" />
                  Foundation
                </Badge>
              )}
            </div>
            {model.description && (
              <p className="text-gray-600 max-w-2xl">{model.description}</p>
            )}
            
            {/* Tags row */}
            <div className="flex flex-wrap gap-2 mt-3">
              {model.license && (
                <Badge className={getLicenseColor(model.license)}>
                  <Scale className="h-3 w-3 mr-1" />
                  {model.license}
                </Badge>
              )}
              {model.task && model.task.length > 0 && (
                model.task.map((task, index) => (
                  <Badge key={index} variant="outline" className="text-blue-700 border-blue-300">
                    <Tag className="h-3 w-3 mr-1" />
                    {task}
                  </Badge>
                ))
              )}
            </div>
          </div>
          <Badge className={getStatusColor(model.status)}>
            {model.status}
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Info */}
        <div className="lg:col-span-2 space-y-6">
          {/* Model Specifications */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Info className="h-5 w-5" />
                <span>Model Specifications</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-500">Total Parameters</label>
                  <p className="text-lg font-semibold">{formatNumber(model.total_parameters)}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Layer Count</label>
                  <p className="text-lg font-semibold">{model.layer_count || 'N/A'}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Structural Hash</label>
                  <p className="text-sm font-mono bg-gray-100 p-2 rounded">
                    {model.structural_hash || 'N/A'}
                  </p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Model ID</label>
                  <p className="text-sm font-mono bg-gray-100 p-2 rounded">
                    {model.id}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Lineage Information */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <GitBranch className="h-5 w-5" />
                <span>Model Lineage</span>
              </CardTitle>
              <CardDescription>
                Parent-child relationships discovered through weight analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              {model.lineage?.parent || model.lineage?.children?.length > 0 ? (
                <div className="space-y-4">
                  {/* Parent */}
                  {model.lineage.parent && (
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Parent Model</h4>
                      <Link to={`/models/${model.lineage.parent.id}`}>
                        <Card className="hover:shadow-md transition-shadow cursor-pointer">
                          <CardContent className="p-4">
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="font-medium">{model.lineage.parent.name}</p>
                                <p className="text-sm text-gray-500">
                                  {formatNumber(model.lineage.parent.total_parameters)} parameters
                                </p>
                              </div>
                              {model.confidence_score && (
                                <Badge variant="outline">
                                  {(model.confidence_score * 100).toFixed(1)}% confidence
                                </Badge>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      </Link>
                    </div>
                  )}

                  {/* Children */}
                  {model.lineage.children?.length > 0 && (
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">
                        Child Models ({model.lineage.children.length})
                      </h4>
                      <div className="space-y-2">
                        {model.lineage.children.map((child) => (
                          <Link key={child.id} to={`/models/${child.id}`}>
                            <Card className="hover:shadow-md transition-shadow cursor-pointer">
                              <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                  <div>
                                    <p className="font-medium">{child.name}</p>
                                    <p className="text-sm text-gray-500">
                                      {formatNumber(child.total_parameters)} parameters
                                    </p>
                                  </div>
                                  {child.confidence_score && (
                                    <Badge variant="outline">
                                      {(child.confidence_score * 100).toFixed(1)}% confidence
                                    </Badge>
                                  )}
                                </div>
                              </CardContent>
                            </Card>
                          </Link>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <GitBranch className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                  <p>No lineage relationships discovered yet</p>
                  <p className="text-sm">This model appears to be standalone or analysis is still in progress</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Dataset URL */}
          {model.dataset_url && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <LinkIcon className="h-5 w-5" />
                  <span>Dataset</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <a 
                    href={model.dataset_url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800 flex items-center gap-1 truncate max-w-[80%]"
                  >
                    {model.dataset_url}
                    <ExternalLink className="h-4 w-4 flex-shrink-0" />
                  </a>
                  <Badge className={getDatasetVerificationBadge(model.dataset_url_verified).className}>
                    {getDatasetVerificationBadge(model.dataset_url_verified).text}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          )}

          {/* README */}
          {model.readme_uri && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="h-5 w-5" />
                  <span>README</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {readmeLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                  </div>
                ) : readmeContent ? (
                  <div className="prose prose-sm max-w-none">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        code({node, inline, className, children, ...props}) {
                          const match = /language-(\w+)/.exec(className || '');
                          return !inline && match ? (
                            <SyntaxHighlighter
                              style={oneDark}
                              language={match[1]}
                              PreTag="div"
                              {...props}
                            >
                              {String(children).replace(/\n$/, '')}
                            </SyntaxHighlighter>
                          ) : (
                            <code className={className} {...props}>
                              {children}
                            </code>
                          );
                        }
                      }}
                    >
                      {readmeContent}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm">Failed to load README content</p>
                )}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Family Info */}
          {model.family_id && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Users className="h-5 w-5" />
                  <span>Model Family</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Badge variant="outline" className="mb-3">
                  {model.family_id}
                </Badge>
                <p className="text-sm text-gray-600">
                  This model belongs to a family of structurally similar models.
                </p>
              </CardContent>
            </Card>
          )}

          {/* Timestamps */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Clock className="h-5 w-5" />
                <span>Timeline</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <label className="text-sm font-medium text-gray-500">Created</label>
                <p className="text-sm">
                  {new Date(model.created_at).toLocaleString()}
                </p>
              </div>
              {model.processed_at && (
                <div>
                  <label className="text-sm font-medium text-gray-500">Processed</label>
                  <p className="text-sm">
                    {new Date(model.processed_at).toLocaleString()}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {/* Download Button */}
              <Button 
                onClick={handleDownload}
                disabled={downloading}
                className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white shadow-md hover:shadow-lg transition-all duration-200"
              >
                {downloading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4 mr-2" />
                    Download Weights
                  </>
                )}
              </Button>
              
              <Link to="/add-model" className="block">
                <Button variant="outline" className="w-full">
                  <TrendingUp className="h-4 w-4 mr-2" />
                  Add Related Model
                </Button>
              </Link>
              <Link to="/models" className="block">
                <Button variant="outline" className="w-full">
                  <Database className="h-4 w-4 mr-2" />
                  Browse All Models
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

