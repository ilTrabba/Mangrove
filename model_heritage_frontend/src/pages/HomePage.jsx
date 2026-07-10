import { Link } from 'react-router-dom';
import { Database, GitBranch, Search, Upload, TrendingUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export default function HomePage() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <div className="flex justify-center mb-6">
          <div className="p-3 bg-blue-100 rounded-full">
            <Database className="h-12 w-12 text-blue-600" />
          </div>
        </div>
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Model Heritage
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
          Automatically discover the lineage and relationships between your machine learning models 
          through advanced weight analysis and genealogy reconstruction.
        </p>
        <div className="flex justify-center space-x-4">
          <Link to="/models">
            <Button size="lg">
              <Search className="h-5 w-5 mr-2" />
              Browse Models
            </Button>
          </Link>
          <Link to="/add-model">
            <Button size="lg" variant="outline">
              <Upload className="h-5 w-5 mr-2" />
              Upload Model
            </Button>
          </Link>
        </div>
      </div>

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
        <Card>
          <CardHeader>
            <GitBranch className="h-8 w-8 text-blue-600 mb-2" />
            <CardTitle>Automatic Lineage Discovery</CardTitle>
            <CardDescription>
              Discover parent-child relationships between models using weight-only analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">
              The MoTHer algorithm analyzes model weights to reconstruct genealogy trees, 
              identifying which models were fine-tuned from others.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <Database className="h-8 w-8 text-green-600 mb-2" />
            <CardTitle>Model Lake Management</CardTitle>
            <CardDescription>
              Organize and catalog your models with intelligent family grouping
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">
              Models are automatically grouped into families based on structural similarity, 
              making it easy to navigate and understand your model ecosystem.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <TrendingUp className="h-8 w-8 text-purple-600 mb-2" />
            <CardTitle>Weight-Only Analysis</CardTitle>
            <CardDescription>
              No metadata required - analysis based purely on model weights
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">
              Unlike traditional approaches, our system works without relying on naming conventions 
              or metadata, providing robust and reliable lineage detection.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* How It Works */}
      <div className="bg-gray-50 rounded-lg p-8 mb-16">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-blue-600 font-bold">1</span>
            </div>
            <h3 className="font-semibold mb-2">Upload Model</h3>
            <p className="text-sm text-gray-600">
              Upload your model file (.safetensors, .pt, .bin)
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-blue-600 font-bold">2</span>
            </div>
            <h3 className="font-semibold mb-2">Extract Signature</h3>
            <p className="text-sm text-gray-600">
              Analyze weights to extract architectural signature
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-blue-600 font-bold">3</span>
            </div>
            <h3 className="font-semibold mb-2">Find Family</h3>
            <p className="text-sm text-gray-600">
              Assign to model family based on structural similarity
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-blue-600 font-bold">4</span>
            </div>
            <h3 className="font-semibold mb-2">Discover Lineage</h3>
            <p className="text-sm text-gray-600">
              Identify parent-child relationships within family
            </p>
          </div>
        </div>
      </div>

      {/* CTA */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Ready to explore your model lineage?
        </h2>
        <p className="text-gray-600 mb-6">
          Start by uploading your first model or browse existing ones.
        </p>
        <div className="flex justify-center space-x-4">
          <Link to="/add-model">
            <Button size="lg">
              Get Started
            </Button>
          </Link>
          <Link to="/models">
            <Button size="lg" variant="outline">
              View Demo
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}

