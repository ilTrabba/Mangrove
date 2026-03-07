import { Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Plus, Leaf, Search } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();

  return (
    <nav className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <div className="p-1.5 bg-emerald-100 rounded-full">
                <Leaf className="h-5 w-5 text-emerald-500" />
              </div>
              <span
                className="text-xl font-bold text-emerald-600"
                style={{ fontFamily: "'Quicksand', sans-serif", letterSpacing: '0.08em' }}
              >
                MANGROVE
              </span>
            </Link>
          </div>
          
          <div className="flex items-center space-x-4">
            <Link
              to="/models"
              className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                location.pathname === '/models'
                  ? 'bg-sky-100 text-sky-700'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
              }`}
            >
              Models
            </Link>

            <Link
              to="/explore"
              className={`px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-1 ${
                location.pathname === '/explore'
                  ?  'bg-sky-100 text-sky-700'
                  :  'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Search className="h-4 w-4" />
              Explore
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