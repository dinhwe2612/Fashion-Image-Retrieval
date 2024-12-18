import React, { useState } from "react";
import axios from "axios";

const FashionSearch = () => {
  const [topK, setTopK] = useState(10); // Default to 10 results
  const [previewImage, setPreviewImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [expandedImage, setExpandedImage] = useState(null);
  const [model, setModel] = useState("FashionIQ");
  const [keyword, setKeyword] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false); // New loading state

  const API_URL = "http://localhost:8000/predict/";

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setPreviewImage(URL.createObjectURL(file));
      setImageFile(file);
    }
  };

  const handleSearch = async () => {
    if (!imageFile && !keyword) {
      alert("Please upload an image or enter a keyword.");
      return;
    }
  
    const formData = new FormData();
  
    try {
      setLoading(true);
      setSearchResults([]);
      
      if (imageFile) formData.append("file", imageFile);
      formData.append("text", keyword);
      formData.append("k", topK); // Add Top K value to request
  
      const response = await axios.post(API_URL, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
  
      const results = response.data;
      setSearchResults(results.top_images);
      console.log("Search Results:", results.top_images);
    } catch (error) {
      console.error("Error contacting backend:", error);
      alert("An error occurred while fetching results.");
    } finally {
      setLoading(false);
    }
  };
  
  const removeImage = () => {
    setPreviewImage(null);
    setImageFile(null);
    document.getElementById("fileUpload").value = "";
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <header className="flex justify-between items-center bg-white px-6 py-4 shadow-md">
        <h1 className="font-bold text-lg">🛍️ Fashion Image Retrieval</h1>
        <div className="relative w-1/3">
          <input
            type="text"
            placeholder="Search"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            className="w-full border border-gray-300 rounded-lg py-2 pl-4 pr-10 focus:outline-none"
          />
          <button
            onClick={handleSearch}
            className="bg-gray-200 rounded-lg absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700"
          >
            🔍
          </button>
        </div>
        <div className="flex justify-end items-center space-x-2">
          {/* <label htmlFor="top-k" className="text-gray-700 font-semibold">Top K</label> */}
          <select
            id="top-k"
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            className="border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
          >
            <option value={5}>Top 5</option>
            <option value={10}>Top 10</option>
            <option value={20}>Top 20</option>
            <option value={50}>Top 50</option>
          </select>
        </div>
        <select
          className="border border-gray-300 rounded-lg px-3 py-2 focus:ring"
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="FashionIQ">FashionIQ</option>
          <option value="Fashion200K">Fashion200K</option>
        </select>
      </header>

      {/* Media Upload Section */}
      <div className="justify-center p-4 bg-white shadow-md flex items-center space-x-4">
        <div className="relative w-40 h-40 border-2 border-gray-200 border-dashed rounded-lg flex items-center justify-center">
          {previewImage ? (
            <>
              <img
                src={previewImage}
                alt="Preview"
                className="w-full h-full object-cover rounded-lg cursor-pointer"
                onClick={() => setExpandedImage(previewImage)}
              />
              <button
                onClick={removeImage}
                className="absolute -top-2 -right-2 bg-red-500 text-white w-6 h-6 flex items-center justify-center rounded-full hover:bg-red-600"
              >
                &times;
              </button>
            </>
          ) : (
            <p className="text-gray-500">No image uploaded</p>
          )}
        </div>
        <input
          type="file"
          onChange={handleImageUpload}
          className="hidden"
          id="fileUpload"
        />
        <label
          htmlFor="fileUpload"
          className="text-blue-500 hover:underline cursor-pointer"
        >
          Upload Image 📷
        </label>
      </div>
      {/* Results Section */}
      <main className="flex-grow p-4 bg-white shadow-md">
        <h2 className="text-xl font-semibold mb-4">
          {keyword && <>Results for <span className="text-blue-500">"{keyword}"</span></>}
        </h2>

        {/* Loading Spinner */}
        {loading ? (
          <div className="flex justify-center items-center h-40">
            <div className="animate-spin rounded-full h-10 w-10 border-t-4 border-blue-500"></div>
            <p className="ml-4 text-gray-500">Fetching results...</p>
          </div>
        ) : (
          <div className="grid grid-cols-4 gap-4">
            {searchResults.length > 0 ? (
              searchResults.map((item, index) => {
                const [imageName, score] = item;
                return (
                  <div
                    key={index}
                    className="relative border rounded-lg overflow-hidden shadow hover:shadow-lg cursor-pointer"
                    onClick={() =>
                      setExpandedImage(`http://localhost:8000/images/${imageName}`)
                    }
                  >
                    <img
                      src={`http://localhost:8000/images/${imageName}`}
                      alt={`Result ${index + 1}`}
                      className="w-full h-full object-cover"
                    />
                    {/* <div className="p-2 text-center">
                      <p className="font-semibold">Score: {score.toFixed(2)}</p>
                    </div> */}
                  </div>
                );
              })
            ) : (
              // <p className="text-gray-500 col-span-4">
              //   Search for your fashion match by uploading an image or entering text.
              // </p>
              <></>
            )}
          </div>
        )}
      </main>

      {/* Expanded Image Modal */}
      {expandedImage && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white p-4 rounded-lg shadow-lg w-1/2 h-1/2 relative">
            <img
              src={expandedImage}
              alt="Expanded"
              className="w-full h-full object-contain rounded-lg"
            />
            <button
              onClick={() => setExpandedImage(null)}
              className="absolute -top-3 -right-3 bg-red-500 text-white w-8 h-8 flex items-center justify-center rounded-full hover:bg-red-600"
            >
              &times;
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FashionSearch;
