import React, { useState, useRef } from "react";

const AddNewReference = ({ onClose }) => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const fileInputRef = useRef(null); // Use a ref for the file input

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    setUploadedFiles((prev) => [...prev, ...files]);
  };

  const handleRemoveFile = (indexToRemove) => {
    setUploadedFiles((prev) => {
      const newFiles = prev.filter((_, index) => index !== indexToRemove);
      
      if (newFiles.length === 0) {
        // If no files are left, clear the file input field
        if (fileInputRef.current) {
          fileInputRef.current.value = ""; // Clear the input field
        }
      }
      
      return newFiles;
    });
  };

  const handleClearInput = (inputId) => {
    const inputElement = document.getElementById(inputId);
    if (inputElement) inputElement.value = ""; // Clear the file input field
  };

  return (
    <div className="fixed inset-0 bg-white bg-opacity-50 flex items-center justify-center z-50 shadow-lg">
      <div className="bg-white p-6 rounded-lg w-3/4 max-h-[80vh] overflow-hidden">
        <h2 className="text-xl font-bold mb-4">Upload Images</h2>

        {/* File Upload Input */}
        <div className="mb-4">
          <input
            id="fileUpload"
            ref={fileInputRef} // Attach ref to the file input
            accept="image/*"
            type="file"
            multiple
            onChange={(e) => {
              handleFileUpload(e);
              handleClearInput("fileUpload"); // Clear input value after adding files
            }}
            className="block w-full mb-2"
          />
          <span className="text-gray-600">{uploadedFiles.length} files</span>
        </div>

        {/* Uploaded Images Display */}
        <div className="border rounded-lg p-4 bg-gray-100 mb-4 max-h-[50vh] overflow-y-auto">
          {uploadedFiles.length > 0 ? (
            <div className="grid grid-cols-4 gap-4">
              {uploadedFiles.map((file, index) => (
                <div key={index} className="relative">
                  <img
                    src={URL.createObjectURL(file)} // Create a URL for the image
                    alt={`Uploaded ${index + 1}`}
                    className="w-full h-auto object-cover rounded-lg shadow"
                  />
                  {/* Remove Image Button */}
                  <button
                    onClick={() => handleRemoveFile(index)} // Remove the selected file
                    className="absolute top-1 right-1 bg-red-500 text-white w-6 h-6 rounded-full flex items-center justify-center hover:bg-red-600"
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500">No images uploaded yet.</p>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end">
          <button
            onClick={() => onClose([])} // Pass empty array on Cancel
            className="bg-gray-500 text-white px-4 py-2 rounded-lg mr-2"
          >
            Cancel
          </button>
          <button
            onClick={() => onClose(uploadedFiles)} // Pass uploaded files on Upload
            className="bg-blue-500 text-white px-4 py-2 rounded-lg"
          >
            Upload
          </button>
        </div>
      </div>
    </div>
  );
};

export default AddNewReference;
