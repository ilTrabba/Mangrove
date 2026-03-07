import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileText, AlertCircle, CheckCircle, Trash2, Eye, EyeOff, X, ChevronDown } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';

const LICENSE_OPTIONS = [
  { value: '', label: 'Select a license.. .' },
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
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    files: [],
    license: '',
    customLicense: '',
    tasks: [],
    datasetUrl: '',
    readmeFile: null,
    isFoundationModel:  false
  });
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [showFoundationModel, setShowFoundationModel] = useState(false);
  const [datasetUrlError, setDatasetUrlError] = useState(null);
  const [showTaskDropdown, setShowTaskDropdown] = useState(false);
  const [showLicenseDropdown, setShowLicenseDropdown] = useState(false);

  const handleFilesChange = (e) => {
    const filesArray = Array.from(e.target.files);
    
    if (filesArray.length === 0) return;
    
    // Validazione client-side per pattern sharded (opzionale, nice-to-have)
    const shardedPattern = /-\d+-of-\d+\. safetensors$/i;
    const hasSharded = filesArray.some(f => shardedPattern.test(f.name));
    
    if (hasSharded && filesArray.length > 1) {
      // Verifica che tutti i file siano safetensors se sembrano sharded
      const allSafetensors = filesArray. every(f => f.name. endsWith('.safetensors'));
      if (!allSafetensors) {
        setError('When uploading sharded files, all files must be .safetensors format');
        return;
      }
    }
    
    setFormData(prev => ({
      ...prev,
      files: filesArray,
      name: prev.name || filesArray[0].name. replace(/\.[^/.]+$/, '')
    }));
    
    setError(null);
  };

  const handleReadmeFileChange = (e) => {
    const file = e. target.files[0];
    if (file) {
      const ext = file.name.split('.').pop().toLowerCase();
      if (!['md', 'txt']. includes(ext)) {
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
      ... prev,
      [name]: value
    }));
  };

  const handleDatasetUrlChange = (e) => {
    const value = e.target.value;
    setFormData(prev => ({
      ...prev,
      datasetUrl: value
    }));
    
    // Validate URL
    if (value && !URL_REGEX. test(value)) {
      setDatasetUrlError('Please enter a valid URL');
    } else {
      setDatasetUrlError(null);
    }
  };

  const handleTaskToggle = (task) => {
    setFormData(prev => {
      const tasks = prev.tasks.includes(task)
        ? prev.tasks. filter(t => t !== task)
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
      files:  [],
      license: '',
      customLicense: '',
      tasks: [],
      datasetUrl:  '',
      readmeFile: null,
      isFoundationModel:  false
    });
    setError(null);
    setSuccess(null);
    setDatasetUrlError(null);
    setShowFoundationModel(false);
    // Reset anche l'input file
    const fileInput = document.getElementById('file');
    const readmeInput = document.getElementById('readme-file');
    if (fileInput) fileInput.value = '';
    if (readmeInput) readmeInput.value = '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.files || formData.files.length === 0) {  
      setError('Please select at least one file to upload');
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
      const formDataToSend = new FormData();

      // Append all files with the same key 'file'
      formData.files.forEach(file => {  
        formDataToSend. append('file', file);
      });

      formDataToSend.append('name', formData.name || formData.files[0].name);
      formDataToSend. append('description', formData.description);
      
      // New fields
      const licenseValue = formData.license === 'Other' ? formData.customLicense : formData.license;
      if (licenseValue) formDataToSend.append('license', licenseValue);
      
      if (formData.tasks. length > 0) {
        formDataToSend.append('task', formData.tasks.join(','));
      }
      
      if (formData. datasetUrl) {
        formDataToSend.append('dataset_url', formData.datasetUrl);
      }
      
      if (formData. readmeFile) {
        formDataToSend.append('readme_file', formData.readmeFile);
      }
      
      formDataToSend.append('is_foundation_model', formData.isFoundationModel. toString());

      const response = await fetch('http://localhost:5001/api/models', {
        method: 'POST',
        body: formDataToSend
      });

      const result = await response.json();

      if (! response.ok) {
        throw new Error(result.error || 'Upload failed');
      }

      setSuccess(`Model "${result.model. name}" uploaded successfully!`);
      
      // Redirect to model detail page after a short delay
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
    const i = Math.floor(Math. log(bytes) / Math.log(k));
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
          <CardTitle>Model Upload</CardTitle>
          <CardDescription>
            Supported formats: .safetensors, .pt, .bin, .pth, .zip, .ckpt
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* File Upload */}
            <div className="space-y-2">
              <Label htmlFor="file">Model File(s) *</Label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-colors">
                <input
                  id="file"
                  type="file"
                  multiple
                  accept=".safetensors,.pt,.bin,.pth,.zip,.ckpt"
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
                  <p className="text-xs text-gray-400 mt-1">
                    Multiple files supported (Ctrl+Click to select)
                  </p>
                </label>
              </div>
              
              {/* Display selected files */}
              {formData.files && formData.files.length > 0 && (
                <div className="space-y-2 mt-3">
                  {formData. files.length === 1 ?  (
                    // Single file display
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
                        title="Remove file"
                      >
                        <Trash2 className="h-7 w-7" />
                      </Button>
                    </div>
                  ) : (
                    // Multiple files display
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
                        {formData. files.map((file, index) => (
                          <div 
                            key={index} 
                            className="flex items-center space-x-2 text-xs text-gray-600 bg-white p-2 rounded"
                          >
                            <FileText className="h-3 w-3 flex-shrink-0" />
                            <span className="flex-1 truncate">{file.name}</span>
                            <span className="text-gray-400 flex-shrink-0">
                              {formatFileSize(file.size)}
                            </span>
                          </div>
                        ))}
                      </div>
                      
                      <p className="text-xs text-gray-500 mt-2">
                        Total size: {formatFileSize(formData.files.reduce((acc, f) => acc + f.size, 0))}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Model Name */}
            <div className="space-y-2">
              <Label htmlFor="name">Model Name *</Label>
              <Input
                id="name"
                name="name"
                type="text"
                value={formData.name}
                onChange={handleInputChange}
                placeholder="Enter a descriptive name for your model"
                required
              />
            </div>

            {/* Description */}
            <div className="space-y-2">
              <Label htmlFor="description">Description (Optional)</Label>
              <Textarea
                id="description"
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                placeholder="Describe your model, its purpose, training details, etc."
                rows={4}
              />
            </div>

            {/* License */}
            <div className="space-y-2">
              <Label htmlFor="license">License (Optional)</Label>
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setShowLicenseDropdown(!showLicenseDropdown)}
                  className="flex h-10 w-full items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                >
                  <span className={formData.license ?  'text-gray-900' : 'text-gray-500'}>
                    {formData.license 
                      ? LICENSE_OPTIONS.find(opt => opt.value === formData.license)?. label || formData.license
                      : 'Select a license.. .'
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
                          formData.license === option.value ?  'bg-blue-50' : ''
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
                  id="customLicense"
                  name="customLicense"
                  type="text"
                  value={formData.customLicense}
                  onChange={handleInputChange}
                  placeholder="Enter custom license name"
                  className="mt-2"
                />
              )}
            </div>

            {/* Tasks Multi-Select */}
            <div className="space-y-2">
              <Label>Tasks (Optional)</Label>
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setShowTaskDropdown(!showTaskDropdown)}
                  className="flex h-10 w-full items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
                >
                  <span className="text-gray-500">
                    {formData.tasks.length === 0 
                      ? 'Select tasks...' 
                      : `${formData.tasks.length} task(s) selected`
                    }
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
                          formData.tasks.includes(task) ?  'bg-blue-50' : ''
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={formData.tasks.includes(task)}
                          readOnly
                          className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 mr-2"
                        />
                        <span className="text-sm">{task}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              {/* Selected Tasks as Badges */}
              {formData. tasks.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-2">
                  {formData.tasks.map(task => (
                    <Badge key={task} variant="secondary" className="flex items-center gap-1 px-2 py-1">
                      {task}
                      <button
                        type="button"
                        onClick={() => removeTask(task)}
                        className="ml-1 hover:text-red-600"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              )}
            </div>

            {/* Dataset URL */}
            <div className="space-y-2">
              <Label htmlFor="datasetUrl">Dataset URL (Optional)</Label>
              <div className="relative">
                <Input
                  id="datasetUrl"
                  name="datasetUrl"
                  type="url"
                  value={formData.datasetUrl}
                  onChange={handleDatasetUrlChange}
                  placeholder="https://huggingface.co/datasets/..."
                  className={datasetUrlError ? 'border-red-500 focus-visible:ring-red-500' : formData.datasetUrl && !datasetUrlError ? 'border-green-500 focus-visible:ring-green-500' : ''}
                />
                {formData.datasetUrl && (
                  <div className="absolute right-3 top-1/2 -translate-y-1/2">
                    {datasetUrlError ?  (
                      <AlertCircle className="h-4 w-4 text-red-500" />
                    ) : (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    )}
                  </div>
                )}
              </div>
              {datasetUrlError && (
                <p className="text-xs text-red-500">{datasetUrlError}</p>
              )}
              <p className="text-xs text-gray-500">Link to dataset on HuggingFace, GitHub, or Kaggle</p>
            </div>

            {/* README File Upload */}
            <div className="space-y-2">
              <Label htmlFor="readme-file">README File (Optional)</Label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-gray-400 transition-colors">
                <input
                  id="readme-file"
                  type="file"
                  accept=".md,.txt"
                  onChange={handleReadmeFileChange}
                  className="hidden"
                />
                <label htmlFor="readme-file" className="cursor-pointer">
                  <FileText className="mx-auto h-8 w-8 text-gray-400 mb-2" />
                  <p className="text-sm text-gray-600">
                    Click to upload README
                  </p>
                  <p className="text-xs text-gray-500">
                    . md or .txt files, max 5MB
                  </p>
                </label>
              </div>
              
              {formData.readmeFile && (
                <div className="flex items-center justify-between text-sm text-gray-600 bg-gray-50 p-2 rounded">
                  <div className="flex items-center space-x-2">
                    <FileText className="h-4 w-4" />
                    <span>{formData.readmeFile. name}</span>
                    <span className="text-gray-400">({formatFileSize(formData.readmeFile.size)})</span>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setFormData(prev => ({ ...prev, readmeFile: null }));
                      const input = document.getElementById('readme-file');
                      if (input) input.value = '';
                    }}
                    className="h-10 w-10 p-0 hover:bg-red-50 hover:text-red-600"
                    title="Remove file"
                  >
                    <Trash2 className="h-7 w-7" />
                  </Button>
                </div>
              )}
            </div>

            {/* Foundation Model Toggle */}
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <button
                  type="button"
                  onClick={() => setShowFoundationModel(!showFoundationModel)}
                  className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
                  title="This field indicates if the model is a foundation/base model"
                >
                  {showFoundationModel ? (
                    <Eye className="h-5 w-5 text-blue-600" />
                  ) : (
                    <EyeOff className="h-5 w-5" />
                  )}
                  <span className="text-sm font-medium">Foundation Model</span>
                </button>
              </div>
              
              <div 
                className={`overflow-hidden transition-all duration-300 ease-in-out ${
                  showFoundationModel ? 'max-h-20 opacity-100' : 'max-h-0 opacity-0'
                }`}
              >
                <div className="flex items-center space-x-2 pt-2">
                  <input
                    type="checkbox"
                    id="isFoundationModel"
                    checked={formData.isFoundationModel}
                    onChange={(e) => setFormData(prev => ({ ...prev, isFoundationModel: e.target.checked }))}
                    className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <Label htmlFor="isFoundationModel" className="text-sm font-normal">
                    This is a foundation/base model
                  </Label>
                </div>
              </div>
            </div>

            {/* Error Alert */}
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Success Alert */}
            {success && (
              <Alert className="border-green-200 bg-green-50">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <AlertDescription className="text-green-800">{success}</AlertDescription>
              </Alert>
            )}

            {/* Submit Button */}
            <div className="flex space-x-4">
              <Button
                type="submit"
                disabled={uploading || !formData.files || formData.files.length === 0}
                className="flex-1"
              >
                {uploading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Processing...
                  </>
                ) : (
                  'Upload Model'
                )}
              </Button>
              
              <Button
                type="button"
                variant="outline"
                onClick={() => navigate('/models')}
                disabled={uploading}
              >
                Cancel
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Info Card */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="text-lg">What happens next?</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-gray-600 space-y-2">
          <p>• Your model will be analyzed to extract its architectural signature</p>
          <p>• The system will automatically assign it to a model family</p>
          <p>• Parent-child relationships will be discovered using weight analysis</p>
          <p>• You'll be able to explore the model's lineage and relationships</p>
        </CardContent>
      </Card>
    </div>
  );
}