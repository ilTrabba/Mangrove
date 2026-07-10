import { Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Plus, Database } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();

  return (
    <nav className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <Database className="h-8 w-8 text-blue-600" />
              <span className="text-xl font-bold text-gray-900">Model Heritage</span>
            </Link>
          </div>
          
          <div className="flex items-center space-x-4">
            <Link
              to="/models"
              className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                location.pathname === '/models'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
              }`}
            >
              Models
            </Link>
            
            <Link to="/add-model">
              <Button className="flex items-center space-x-2">
                <Plus className="h-4 w-4" />
                <span>Add Model</span>
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

