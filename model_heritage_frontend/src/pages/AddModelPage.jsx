import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileText, AlertCircle, CheckCircle, Trash2, Eye, EyeOff, X, ChevronDown, Server, Laptop } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';

const LICENSE_OPTIONS = [
  { value: '', label: 'Select a license...' },
  { value: 'MIT', label: 'MIT' },
  { value: 'Apache-2.0', label: 'Apache 2.0' },
  { value: 'GPL-3.0', label: 'GPL-3.0' },
  { value: 'BSD-3-Clause', label: 'BSD-3-Clause' },
  { value: 'CC-BY-NC-4.0', label: 'CC BY-NC 4.0' },
  { value: 'Proprietary', label: 'Proprietary' },
  { value: 'Other', label: 'Other' }
];

const TASK_OPTIONS = [
  'Text Generation',
  'Image Classification', 
  'Object Detection',
  'Text Classification',
  'Question Answering',
  'Translation',
  'Summarization',
  'Image-to-Text',
  'Text-to-Image',
  'Other'
];

// URL validation regex
const URL_REGEX = /^https?:\/\/(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|localhost|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::\d+)?(?:\/?|[/?]\S+)$/i;

export default function AddModelPage() {
  const navigate = useNavigate();
  
  // --- STATE ---
  const [uploadMode, setUploadMode] = useState('local'); // 'local' | 'vm'
  const [vmFiles, setVmFiles] = useState([]);
  const [selectedVmPath, setSelectedVmPath] = useState('');
  const [isLoadingVmFiles, setIsLoadingVmFiles] = useState(false);

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    files: [],
    license: '',
    customLicense: '',
    tasks: [],
    datasetUrl: '',
    readmeFile: null,
    isFoundationModel: false
  });

  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [showFoundationModel, setShowFoundationModel] = useState(false);
  const [datasetUrlError, setDatasetUrlError] = useState(null);
  const [showTaskDropdown, setShowTaskDropdown] = useState(false);
  const [showLicenseDropdown, setShowLicenseDropdown] = useState(false);

  // --- EFFECTS ---

  // Fetch VM files when switching to VM mode
  useEffect(() => {
    if (uploadMode === 'vm') {
      setIsLoadingVmFiles(true);
      setError(null);

      // ATTENZIONE ALL'URL: Verifica se il tuo blueprint è sotto /api/models o solo /api
      // Se fallisce con 404, prova: 'http://localhost:5001/api/vm-files'
      fetch('http://localhost:5002/api/vm-files')
        .then(res => {
          if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
          return res.json();
        })
        .then(data => {
          setVmFiles(data.files || []);
          setIsLoadingVmFiles(false);
        })
        .catch(err => {
          console.error("Error fetching VM files:", err);
          setError(`Could not load files from VM: ${err.message}. Is Backend running?`);
          setIsLoadingVmFiles(false); // CRUCIALE: Ferma lo spinner anche se fallisce
        });
    }
  }, [uploadMode]);

  // --- HANDLERS ---

  const handleFilesChange = (e) => {
    const filesArray = Array.from(e.target.files);
    
    if (filesArray.length === 0) return;
    
    const shardedPattern = /-\d+-of-\d+\.safetensors$/i;
    const hasSharded = filesArray.some(f => shardedPattern.test(f.name));
    
    if (hasSharded && filesArray.length > 1) {
      const allSafetensors = filesArray.every(f => f.name.endsWith('.safetensors'));
      if (!allSafetensors) {
        setError('When uploading sharded files, all files must be .safetensors format');
        return;
      }
    }
    
    setFormData(prev => ({
      ...prev,
      files: filesArray,
      name: prev.name || filesArray[0].name.replace(/\.[^/.]+$/, '')
    }));
    
    setError(null);
  };

  const handleVmFileSelect = (e) => {
    const path = e.target.value;
    setSelectedVmPath(path);
    
    // Auto-fill name based on selected VM file
    if (path) {
      const selectedFile = vmFiles.find(f => f.path === path);
      if (selectedFile && !formData.name) {
        setFormData(prev => ({
          ...prev,
          name: selectedFile.name.replace(/\.[^/.]+$/, '')
        }));
      }
    }
  };

  const handleReadmeFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const ext = file.name.split('.').pop().toLowerCase();
      if (!['md', 'txt'].includes(ext)) {
        setError('README file must be .md or .txt');
        return;
      }
      if (file.size > 5 * 1024 * 1024) {
        setError('README file must be less than 5MB');
        return;
      }
      setFormData(prev => ({
        ...prev,
        readmeFile: file
      }));
      setError(null);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleDatasetUrlChange = (e) => {
    const value = e.target.value;
    setFormData(prev => ({
      ...prev,
      datasetUrl: value
    }));
    
    if (value && !URL_REGEX.test(value)) {
      setDatasetUrlError('Please enter a valid URL');
    } else {
      setDatasetUrlError(null);
    }
  };

  const handleTaskToggle = (task) => {
    setFormData(prev => {
      const tasks = prev.tasks.includes(task)
        ? prev.tasks.filter(t => t !== task)
        : [...prev.tasks, task];
      return { ...prev, tasks };
    });
  };

  const removeTask = (task) => {
    setFormData(prev => ({
      ...prev,
      tasks: prev.tasks.filter(t => t !== task)
    }));
  };

  const handleReset = () => {
    setFormData({
      name: '',
      description: '',
      files: [],
      license: '',
      customLicense: '',
      tasks: [],
      datasetUrl: '',
      readmeFile: null,
      isFoundationModel: false
    });
    setSelectedVmPath('');
    setError(null);
    setSuccess(null);
    setDatasetUrlError(null);
    setShowFoundationModel(false);
    
    const fileInput = document.getElementById('file');
    const readmeInput = document.getElementById('readme-file');
    if (fileInput) fileInput.value = '';
    if (readmeInput) readmeInput.value = '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validation based on mode
    if (uploadMode === 'local' && (!formData.files || formData.files.length === 0)) {
      setError('Please select at least one file to upload');
      return;
    }
    if (uploadMode === 'vm' && !selectedVmPath) {
      setError('Please select a file from the VM list');
      return;
    }

    if (datasetUrlError) {
      setError('Please fix the dataset URL before submitting');
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(null);

    try {
      let response;
      
      // === MODE 1: LOCAL UPLOAD (FormData) ===
      if (uploadMode === 'local') {
        const formDataToSend = new FormData();
        formData.files.forEach(file => {
          formDataToSend.append('file', file);
        });
        
        // Append fields
        formDataToSend.append('name', formData.name || formData.files[0].name);
        formDataToSend.append('description', formData.description);
        const licenseValue = formData.license === 'Other' ? formData.customLicense : formData.license;
        if (licenseValue) formDataToSend.append('license', licenseValue);
        if (formData.tasks.length > 0) formDataToSend.append('task', formData.tasks.join(','));
        if (formData.datasetUrl) formDataToSend.append('dataset_url', formData.datasetUrl);
        if (formData.readmeFile) formDataToSend.append('readme_file', formData.readmeFile);
        formDataToSend.append('is_foundation_model', formData.isFoundationModel.toString());

        response = await fetch('http://localhost:5002/api/models', {
          method: 'POST',
          body: formDataToSend
        });
      } 
      
      // === MODE 2: VM SELECTION (JSON) ===
      else {
        const payload = {
          local_vm_path: selectedVmPath,
          name: formData.name,
          description: formData.description,
          license: formData.license === 'Other' ? formData.customLicense : formData.license,
          task: formData.tasks.join(','),
          dataset_url: formData.datasetUrl,
          is_foundation_model: formData.isFoundationModel.toString()
        };

        response = await fetch('http://localhost:5002/api/models', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        });
      }

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Processing failed');
      }

      setSuccess(`Model "${result.model.name}" processed successfully!`);
      
      setTimeout(() => {
        navigate(`/models/${result.model.id}`);
      }, 2000);

    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Add New Model</h1>
        <p className="text-gray-600">
          Upload a machine learning model to automatically discover its lineage and relationships.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Source Selection</CardTitle>
          <CardDescription>
            Choose where the model file is located.
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            
            {/* --- MODE SWITCHER --- */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div 
                onClick={() => setUploadMode('local')}
                className={`cursor-pointer border rounded-lg p-4 flex flex-col items-center justify-center transition-all ${
                  uploadMode === 'local' 
                    ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <Laptop className={`h-8 w-8 mb-2 ${uploadMode === 'local' ? 'text-blue-600' : 'text-gray-400'}`} />
                <span className={`font-medium ${uploadMode === 'local' ? 'text-blue-700' : 'text-gray-600'}`}>
                  Upload from PC
                </span>
              </div>

              <div 
                onClick={() => setUploadMode('vm')}
                className={`cursor-pointer border rounded-lg p-4 flex flex-col items-center justify-center transition-all ${
                  uploadMode === 'vm' 
                    ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <Server className={`h-8 w-8 mb-2 ${uploadMode === 'vm' ? 'text-blue-600' : 'text-gray-400'}`} />
                <span className={`font-medium ${uploadMode === 'vm' ? 'text-blue-700' : 'text-gray-600'}`}>
                  Select from VM
                </span>
              </div>
            </div>

            {/* --- FILE SELECTION AREA --- */}
            <div className="space-y-2">
              <Label htmlFor={uploadMode === 'local' ? 'file' : 'vm-file-select'}>
                {uploadMode === 'local' ? 'Model File(s) *' : 'Select Server File *'}
              </Label>
              
              {uploadMode === 'local' ? (
                // === LOCAL UPLOAD UI ===
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-colors">
                  <input
                    id="file"
                    type="file"
                    multiple
                    accept=".safetensors,.pt,.bin,.pth,.zip"
                    onChange={handleFilesChange}
                    className="hidden"
                  />
                  <label htmlFor="file" className="cursor-pointer">
                    <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                    <p className="text-sm text-gray-600 mb-2">
                      Click to upload or drag and drop
                    </p>
                    <p className="text-xs text-gray-500">
                      SafeTensors, PyTorch, or Pickle files
                    </p>
                  </label>
                </div>
              ) : (
                // === VM SELECTION UI ===
                <div className="space-y-2">
                  {isLoadingVmFiles ? (
                     <div className="flex items-center justify-center p-8 border rounded-lg bg-gray-50">
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mr-3"></div>
                        <span className="text-gray-600">Loading files from VM...</span>
                     </div>
                  ) : (
                    <div className="relative">
                       {/* FIX: Label e ID corretti */}
                       <select
                          id="vm-file-select"
                          name="vmFile"
                          autoComplete="off"
                          value={selectedVmPath}
                          onChange={handleVmFileSelect}
                          className="flex h-12 w-full items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                        >
                          <option value="">-- Select a file available on VM --</option>
                          {vmFiles.map((f, idx) => (
                            <option key={idx} value={f.path}>
                              [{f.folder}] {f.name} ({formatFileSize(f.size)})
                            </option>
                          ))}
                        </select>
                        <p className="text-xs text-gray-500 mt-1">
                          Showing .safetensors and .bin files found in VM project folder.
                        </p>
                    </div>
                  )}
                </div>
              )}

              {/* Display selected files (Local Mode Only) */}
              {uploadMode === 'local' && formData.files && formData.files.length > 0 && (
                <div className="space-y-2 mt-3">
                  {formData.files.length === 1 ? (
                    <div className="flex items-center justify-between space-x-2 text-sm text-gray-600 bg-gray-50 p-3 rounded">
                      <div className="flex items-center space-x-2">
                        <FileText className="h-4 w-4" />
                        <span>{formData.files[0].name}</span>
                        <span className="text-gray-400">({formatFileSize(formData.files[0].size)})</span>
                      </div>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={handleReset}
                        className="h-10 w-10 p-0 hover:bg-red-50 hover:text-red-600"
                      >
                        <Trash2 className="h-7 w-7" />
                      </Button>
                    </div>
                  ) : (
                    <div className="bg-gray-50 p-3 rounded space-y-2">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-medium text-gray-700">
                          {formData.files.length} files selected
                        </p>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={handleReset}
                          className="h-8 px-3 hover:bg-red-50 hover:text-red-600"
                        >
                          <Trash2 className="h-4 w-4 mr-1" />
                          Clear all
                        </Button>
                      </div>
                      <div className="max-h-40 overflow-y-auto space-y-1">
                        {formData.files.map((file, index) => (
                          <div key={index} className="flex items-center space-x-2 text-xs text-gray-600 bg-white p-2 rounded">
                            <FileText className="h-3 w-3 flex-shrink-0" />
                            <span className="flex-1 truncate">{file.name}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* --- REST OF THE FORM (Common Fields) --- */}

            {/* Model Name */}
            <div className="space-y-2">
              <Label htmlFor="name">Model Name *</Label>
              <Input
                id="name"
                name="name"
                type="text"
                value={formData.name}
                onChange={handleInputChange}
                placeholder="Enter a descriptive name"
                required
              />
            </div>

            {/* Description */}
            <div className="space-y-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                placeholder="Describe your model..."
                rows={4}
              />
            </div>

            {/* License */}
            <div className="space-y-2">
              <Label htmlFor="license">License</Label>
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setShowLicenseDropdown(!showLicenseDropdown)}
                  className="flex h-10 w-full items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500"
                >
                  <span className={formData.license ? 'text-gray-900' : 'text-gray-500'}>
                    {formData.license 
                      ? LICENSE_OPTIONS.find(opt => opt.value === formData.license)?.label || formData.license
                      : 'Select a license...'
                    }
                  </span>
                  <ChevronDown className={`h-4 w-4 transition-transform ${showLicenseDropdown ? 'rotate-180' : ''}`} />
                </button>
                
                {showLicenseDropdown && (
                  <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
                    {LICENSE_OPTIONS.filter(opt => opt.value !== '').map(option => (
                      <div
                        key={option.value}
                        onClick={() => {
                          setFormData(prev => ({
                            ...prev,
                            license: option.value,
                            customLicense: option.value !== 'Other' ? '' : prev.customLicense
                          }));
                          setShowLicenseDropdown(false);
                        }}
                        className={`flex items-center px-3 py-2 cursor-pointer hover:bg-gray-100 ${
                          formData.license === option.value ? 'bg-blue-50' : ''
                        }`}
                      >
                        <span className="text-sm">{option.label}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {formData.license === 'Other' && (
                <Input
                  name="customLicense"
                  value={formData.customLicense}
                  onChange={handleInputChange}
                  placeholder="Enter custom license name"
                  className="mt-2"
                />
              )}
            </div>

            {/* Tasks */}
            <div className="space-y-2">
              <Label>Tasks</Label>
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setShowTaskDropdown(!showTaskDropdown)}
                  className="flex h-10 w-full items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500"
                >
                  <span className="text-gray-500">
                    {formData.tasks.length === 0 ? 'Select tasks...' : `${formData.tasks.length} task(s) selected`}
                  </span>
                  <ChevronDown className={`h-4 w-4 transition-transform ${showTaskDropdown ? 'rotate-180' : ''}`} />
                </button>
                
                {showTaskDropdown && (
                  <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
                    {TASK_OPTIONS.map(task => (
                      <div
                        key={task}
                        onClick={() => handleTaskToggle(task)}
                        className={`flex items-center px-3 py-2 cursor-pointer hover:bg-gray-100 ${
                          formData.tasks.includes(task) ? 'bg-blue-50' : ''
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={formData.tasks.includes(task)}
                          readOnly
                          className="h-4 w-4 mr-2"
                        />
                        <span className="text-sm">{task}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {formData.tasks.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-2">
                  {formData.tasks.map(task => (
                    <Badge key={task} variant="secondary" className="flex items-center gap-1 px-2 py-1">
                      {task}
                      <button type="button" onClick={() => removeTask(task)} className="ml-1 hover:text-red-600">
                        <X className="h-3 w-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              )}
            </div>

            {/* Dataset URL */}
            <div className="space-y-2">
              <Label htmlFor="datasetUrl">Dataset URL</Label>
              <div className="relative">
                <Input
                  id="datasetUrl"
                  name="datasetUrl"
                  type="url"
                  value={formData.datasetUrl}
                  onChange={handleDatasetUrlChange}
                  placeholder="https://huggingface.co/datasets/..."
                  className={datasetUrlError ? 'border-red-500' : formData.datasetUrl && !datasetUrlError ? 'border-green-500' : ''}
                />
                {formData.datasetUrl && (
                  <div className="absolute right-3 top-1/2 -translate-y-1/2">
                    {datasetUrlError ? <AlertCircle className="h-4 w-4 text-red-500" /> : <CheckCircle className="h-4 w-4 text-green-500" />}
                  </div>
                )}
              </div>
            </div>

            {/* Readme (Local Mode Only) */}
            {uploadMode === 'local' && (
              <div className="space-y-2">
                <Label htmlFor="readme-file">README File</Label>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-gray-400">
                  <input
                    id="readme-file"
                    type="file"
                    accept=".md,.txt"
                    onChange={handleReadmeFileChange}
                    className="hidden"
                  />
                  <label htmlFor="readme-file" className="cursor-pointer">
                    <FileText className="mx-auto h-8 w-8 text-gray-400 mb-2" />
                    <p className="text-sm text-gray-600">Click to upload README</p>
                  </label>
                </div>
                {formData.readmeFile && (
                   <div className="flex items-center justify-between text-sm text-gray-600 bg-gray-50 p-2 rounded">
                     <span>{formData.readmeFile.name}</span>
                     <Button type="button" variant="ghost" size="sm" onClick={() => setFormData(prev => ({...prev, readmeFile: null}))}>
                       <Trash2 className="h-4 w-4" />
                     </Button>
                   </div>
                )}
              </div>
            )}

            {/* Foundation Model Toggle */}
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <button
                  type="button"
                  onClick={() => setShowFoundationModel(!showFoundationModel)}
                  className="flex items-center space-x-2 text-gray-600"
                >
                  {showFoundationModel ? <Eye className="h-5 w-5 text-blue-600" /> : <EyeOff className="h-5 w-5" />}
                  <span className="text-sm font-medium">Foundation Model</span>
                </button>
              </div>
              <div className={`overflow-hidden transition-all duration-300 ${showFoundationModel ? 'max-h-20 opacity-100' : 'max-h-0 opacity-0'}`}>
                <div className="flex items-center space-x-2 pt-2">
                  <input
                    type="checkbox"
                    id="isFoundationModel"
                    checked={formData.isFoundationModel}
                    onChange={(e) => setFormData(prev => ({ ...prev, isFoundationModel: e.target.checked }))}
                    className="h-4 w-4 text-blue-600 border-gray-300 rounded"
                  />
                  <Label htmlFor="isFoundationModel" className="text-sm font-normal">This is a foundation/base model</Label>
                </div>
              </div>
            </div>

            {/* Alerts */}
            {error && <Alert variant="destructive"><AlertCircle className="h-4 w-4" /><AlertDescription>{error}</AlertDescription></Alert>}
            {success && <Alert className="border-green-200 bg-green-50"><CheckCircle className="h-4 w-4 text-green-600" /><AlertDescription className="text-green-800">{success}</AlertDescription></Alert>}

            {/* Actions */}
            <div className="flex space-x-4">
              <Button type="submit" disabled={uploading} className="flex-1">
                {uploading ? 'Processing...' : (uploadMode === 'local' ? 'Upload Model' : 'Process VM File')}
              </Button>
              <Button type="button" variant="outline" onClick={() => navigate('/models')} disabled={uploading}>Cancel</Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}