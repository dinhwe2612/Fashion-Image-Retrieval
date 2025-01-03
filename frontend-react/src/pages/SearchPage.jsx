import React, { useEffect, useState } from "react";
import { IoMdAdd } from "react-icons/io";
import AddNewReference from "./AddNewReference";
import axios from "axios";
import AddDatasetModal from "./AddNewDataset";

const FashionSearch = () => {
  const [datasets, setDatasets] = useState(["FashionIQ"]);
  const [currentQuery, setCurrentQuery] = useState(null)
  const [isEvaluating, SetIsEvaluating] = useState(false);
  const [topK, setTopK] = useState(10); // Default to 10 results
  const [previewImage, setPreviewImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [expandedImage, setExpandedImage] = useState(null);
  const [model, setModel] = useState("FashionIQ");
  const [keyword, setKeyword] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false); // New loading state
  const [selectedImages, setSelectedImages] = useState([]);
  const [unselectedImages, setUnselectedIamges] = useState([])
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDatasetModalOpen, setIsDatasetModalOpen] = useState(false);

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      handleSearch(); // Trigger search when Enter key is pressed
    }
  };

  const handleAddDataset = async (newDataset) => {
    if (!newDataset || !newDataset.name || !newDataset.files || newDataset.files.length === 0) {
      alert("Please provide a valid dataset name and files.");
      return;
    }
  
    const formData = new FormData();
    formData.append("dataset_name", newDataset.name);
  
    // Append each file to the form data
    newDataset.files.forEach((file) => {
      formData.append("files", file);
    });
  
    try {
      const CREATE_DATASET_URL = "http://localhost:8000/create_dataset/"; 
      const response = await axios.post(CREATE_DATASET_URL, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
  
      if (response.status === 200 && response.data.message) {
        // Add the new dataset to the dropdown and set it as the selected dataset
        setDatasets((prev) => [...prev, newDataset.name]);
        setModel(newDataset.name);
        alert(`Dataset "${newDataset.name}" added successfully!`);
      } else {
        alert("Failed to add the dataset. Please try again.");
      }
    } catch (error) {
      console.error("Error while creating the dataset:", error);
      alert("An error occurred while adding the dataset. Please try again.");
    } finally {
      setIsDatasetModalOpen(false); // Close the modal
    }
  };  

  const handleSelection = (imageName) => {
    setSelectedImages((prev) =>
      prev.includes(imageName)
        ? prev.filter((item) => item !== imageName) // Remove if already selected
        : [...prev, imageName] // Add if not selected
    );
  };

  useEffect(() => {
    // Ensure unselectedImages includes all searchResults if no images are selected
    const unselected = searchResults.length > 0
      ? searchResults
          .map(([imageName]) => imageName) // Extract image names from searchResults
          .filter((imageName) => !selectedImages.includes(imageName)) // Exclude selected images
      : []; // Handle empty searchResults gracefully
  
    setUnselectedIamges(unselected);
  }, [selectedImages, searchResults]); // Depend on both selectedImages and searchResults  

  useEffect(() => {
    setSearchResults([]);
    setCurrentQuery({keyword: keyword, imageFile: imageFile});
  }, [keyword, imageFile]);  

  const blobToBase64 = (blob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };  

  const sendFeedbackToServer = async () => {
    if (selectedImages.length > 0) {
      try {
        // Compute unselected images
        // const unselectedImages = searchResults
        //   .map(([imageName]) => imageName)
        //   .filter((imageName) => !selectedImages.includes(imageName));
  
        // Update the currentQuery to include the feedback iteration
        setCurrentQuery((prevQuery) => {
          const updatedQuery = {
            ...prevQuery,
            feedback: [
              ...(prevQuery?.feedback || []),
              { selected: selectedImages, unselected: unselectedImages },
            ],
          };
          console.log("Updated currentQuery:", updatedQuery);
          return updatedQuery;
        });
  
        // Prepare FormData
        const formData = new FormData();
        formData.append("k", topK);
  
        if (imageFile) {
          formData.append("imageFile", imageFile); // Append image file
        }
  
        formData.append("keyword", keyword || ""); // Append keyword (can be empty)
  
        // Append feedback data as JSON string
        const feedbackPayload = JSON.stringify({
          keyword,
          imageFile: null, // Only the file is sent separately
          feedback: [
            ...(currentQuery?.feedback || []),
            { selected: selectedImages, unselected: unselectedImages },
          ],
        });
        formData.append("current_query", feedbackPayload);
  
        // Set loading state and clear search results
        setLoading(true);
        setSearchResults([]);
  
        // Send feedback to the server
        const response = await axios.post("http://localhost:8000/feedback/", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
  
        // Update the search results with the updated results from the server
        setSearchResults(response.data.updated_results);
  
        // Clear the selected images after sending feedback
        setSelectedImages([]);
        // alert("Feedback successfully submitted!");
      } catch (error) {
        console.error("Error sending feedback:", error);
        alert("Failed to submit feedback. Please try again.");
      } finally {
        setLoading(false);
      }
    } else {
      alert("Please select at least one image to provide feedback.");
    }
  };    

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
      formData.append("dataset", model); // Add Top K value to request
  
      const response = await axios.post(API_URL, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
  
      const results = response.data;
      setSearchResults(results.top_images);
      // console.log("Search Results:", results.top_images);
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
  
  const handleAddRef = async (uploadedFiles) => {
    if (uploadedFiles.length === 0) {
      console.log("No files uploaded.");
      // alert("Please upload at least one file.");
      setIsModalOpen(false)
      return;
    }
  
    const formData = new FormData();
  
    // Append files to FormData
    uploadedFiles.forEach((file) => {
      formData.append("files", file);
    });
  
    // Append dataset name
    formData.append("dataset_name", model); // `model` should represent the currently selected dataset
  
    try {
      const API_URL = "http://localhost:8000/append_to_dataset/";
      const response = await axios.post(API_URL, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
  
      // Handle success
      console.log("API response:", response.data);
      alert(`Successfully added ${response.data.image_count} new references to dataset "${model}"!`);
    } catch (error) {
      // Handle error
      console.error("Error appending references:", error);
      alert("An error occurred while adding references. Please try again.");
    } finally {
      setIsModalOpen(false); // Close modal
    }
  };
  
  return (
    <div className="h-screen w-screen flex flex-col bg-gray-100">
      {/* Header */}
      <header className="flex justify-between items-center bg-white px-6 py-4 shadow-md">
        <h1 className="font-bold text-2xl">🛍️ Fashion Image Retrieval</h1>
        <div className="relative w-1/3">
          <input
            type="text"
            placeholder="Search"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            onKeyPress={handleKeyPress} // Listen for Enter key press
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
        <div>
          <span className="text-gray-700 font-semibold mr-2">Dataset:</span>
          <select
            value={model}
            onChange={(e) => {
              if (e.target.value === "Add") {
                setIsDatasetModalOpen(true);
              } else {
                setModel(e.target.value);
              }
            }}
            className="border border-gray-300 rounded-lg py-2 px-3 focus:ring text-left"
          >
            {datasets.map((dataset, index) => (
              <option key={index} value={dataset}>
                {dataset}
              </option>
            ))}
            <option value="Add">+ Add new dataset</option>
          </select>
        </div>
        {isModalOpen && <AddDatasetModal onClose={handleAddDataset} />}
        <div className="flex items-center">
          <IoMdAdd className="text-gray-700 mr-2" />
          <span 
            className="text-gray-700 font-semibold hover:text-black cursor-pointer"
            onClick={() => setIsModalOpen(true)}
          >
            Add new reference
          </span>
        </div>
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
        <h2 className="text-right text-md mb-4 text-gray-500">
          {searchResults.length > 0 && (
            <>{isEvaluating ? (<> Have selected all the relevant images?</>) : (<> Not satisfied?</>)}  
              {searchResults.length > 0 ?
                (<a onClick={
                  async () => {
                    if (!isEvaluating) {
                      // setSelectedImages([])
                    }
                    else {
                      await sendFeedbackToServer();
                    }
                    SetIsEvaluating(!isEvaluating)
                  }
                } className="text-blue-500 hover:underline cursor-pointer">{
                  isEvaluating ? (<> Done Feedback</>) : (<> Feedback</>)
                }</a>)
                : (<div></div>)
              }
            </>
          )}
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
                const isSelected = selectedImages.includes(imageName);
                return (
                  <div
                    key={index}
                    className="relative border rounded-lg overflow-hidden shadow hover:shadow-lg cursor-pointer"
                    onClick={() =>
                      setExpandedImage(`http://localhost:8000/images/${model}/${imageName}`)
                    }
                  >
                    <img
                      src={`http://localhost:8000/images/${model}/${imageName}`}
                      alt={`Result ${index + 1}`}
                      className="w-full h-full object-cover"
                    />
                    {isEvaluating && (
                      <div
                        className="absolute top-2 right-2 w-6 h-6 bg-white border-2 border-green-300 rounded-full flex items-center justify-center"
                        onClick={(e) => {
                          e.stopPropagation(); 
                          handleSelection(imageName);
                        }}
                      >
                        {isSelected && (
                          <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                        )}
                      </div>
                    )}
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
      {isModalOpen && (
        <AddNewReference onClose={(upload) => {
           console.log("Upload", upload);
           handleAddRef(upload)
        }} />
      )}
      {isDatasetModalOpen && (
        <AddDatasetModal onClose={(upload) => {
          if (upload) {
            console.log("Upload", upload);
            handleAddDataset(upload)
          }
          else {
            setIsDatasetModalOpen(false)
          }
        }} />
      )}
    </div>
  );
};

export default FashionSearch;
