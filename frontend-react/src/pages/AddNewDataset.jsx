import React, { useState, useRef } from "react";

const AddDatasetModal = ({ onClose }) => {
  const [datasetName, setDatasetName] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null); // Ref for the file input

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    setUploadedFiles((prev) => [...prev, ...files]);
  };

  const handleRemoveFile = (indexToRemove) => {
    setUploadedFiles((prev) =>
      prev.filter((_, index) => index !== indexToRemove)
    );
    if (uploadedFiles.length <= 1) {
      // If there are no files left, clear the file input
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = () => {
    if (!datasetName.trim()) {
      setError("Dataset name is required.");
      return;
    }
    if (uploadedFiles.length === 0) {
      setError("Please upload at least one file.");
      return;
    }
    setError("");
    onClose({ name: datasetName, files: uploadedFiles });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded-lg shadow-lg w-3/4 max-h-[80vh] overflow-auto">
        <h2 className="text-xl font-bold mb-4">Add New Dataset</h2>
        <div className="mb-4">
          <label className="block text-gray-700 font-semibold mb-2">
            Dataset Name
          </label>
          <input
            type="text"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            placeholder="Enter dataset name"
            className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring focus:ring-blue-500"
          />
        </div>
        <div className="mb-4">
          <label className="block text-gray-700 font-semibold mb-2">
            Upload Files
          </label>
          <input
            type="file"
            multiple
            ref={fileInputRef} // Attach the ref
            onChange={handleFileUpload}
            className="block w-full"
          />
        </div>
        <div className="grid grid-cols-4 gap-4">
          {uploadedFiles.map((file, index) => (
            <div key={index} className="relative">
              <img
                src={URL.createObjectURL(file)}
                alt={`File ${index + 1}`}
                className="w-full h-auto object-cover rounded-lg"
              />
              <button
                onClick={() => handleRemoveFile(index)}
                className="absolute top-1 right-1 bg-red-500 text-white w-6 h-6 rounded-full flex items-center justify-center hover:bg-red-600"
              >
                &times;
              </button>
            </div>
          ))}
        </div>
        {error && <p className="text-red-500 mt-2">{error}</p>}
        <div className="flex justify-end mt-4">
          <button
            onClick={() => onClose(null)}
            className="bg-gray-500 text-white px-4 py-2 rounded-lg mr-2"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            className="bg-blue-500 text-white px-4 py-2 rounded-lg"
          >
            Add Dataset
          </button>
        </div>
      </div>
    </div>
  );
};

export default AddDatasetModal;
