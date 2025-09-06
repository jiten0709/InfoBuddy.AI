"use client";

import Header from "@/components/Header";
import InputBar from "@/components/InputBar";
import MessageArea from "@/components/MessageArea";
import React, { useState, useEffect } from "react";

// global variables
const serverUrl = process.env.NEXT_PUBLIC_SERVER_URL;
console.log("Using server URL:", serverUrl);

// Quick health check
fetch(`${serverUrl}/health`)
  .then((res) => res.json())
  .then((data) => console.log("Health response:", data))
  .catch((err) => console.error("Health check failed:", err));

interface SearchInfo {
  stages: string[];
  query: string;
  urls: string[];
}

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  type: string;
  isLoading?: boolean;
  searchInfo?: SearchInfo;
}

const Home = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      content: "Hi there, how can I help you?",
      isUser: false,
      type: "message",
    },
  ]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [checkpointId, setCheckpointId] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (currentMessage.trim()) {
      // First add the user message to the chat
      const newMessageId =
        messages.length > 0
          ? Math.max(...messages.map((msg) => msg.id)) + 1
          : 1;

      setMessages((prev) => [
        ...prev,
        {
          id: newMessageId,
          content: currentMessage,
          isUser: true,
          type: "message",
        },
      ]);

      const userInput = currentMessage;
      setCurrentMessage(""); // Clear input field immediately

      try {
        // Create AI response placeholder
        const aiResponseId = newMessageId + 1;
        setMessages((prev) => [
          ...prev,
          {
            id: aiResponseId,
            content: "",
            isUser: false,
            type: "message",
            isLoading: true,
            searchInfo: {
              stages: [],
              query: "",
              urls: [],
            },
          },
        ]);

        let url = `${serverUrl}/chat_stream/${encodeURIComponent(userInput)}`;
        if (checkpointId) {
          url += `?conversation_id=${encodeURIComponent(checkpointId)}`;
        }

        console.log("🔍 Connecting to URL:", url); // Debug log

        // Connect to SSE endpoint using EventSource
        const eventSource = new EventSource(url);
        let streamedContent = "";
        let searchData = null;
        let hasReceivedContent = false;

        // Process incoming messages
        eventSource.onmessage = (event) => {
          console.log("🔍 Raw event data:", event.data); // Debug log

          try {
            // Try to parse the JSON
            const data = JSON.parse(event.data);
            console.log("🔍 Parsed event data:", data); // Debug log

            if (data.type === "checkpoint") {
              console.log("✅ Received checkpoint:", data.checkpoint_id);
              setCheckpointId(data.checkpoint_id);
            } else if (data.type === "message_chunk") {
              streamedContent += data.content;
              hasReceivedContent = true;
              console.log("✅ Received content chunk:", data.content);

              // Update message with accumulated content
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiResponseId
                    ? { ...msg, content: streamedContent, isLoading: false }
                    : msg
                )
              );
            } else if (data.type === "search_start") {
              console.log("✅ Search started:", data.query);
              const newSearchInfo = {
                stages: ["searching"],
                query: data.query,
                urls: [],
              };
              searchData = newSearchInfo;

              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === aiResponseId
                    ? {
                        ...msg,
                        content: streamedContent,
                        searchInfo: newSearchInfo,
                        isLoading: false,
                      }
                    : msg
                )
              );
            } else if (data.type === "search_results") {
              console.log("✅ Search results received:", data.urls);
              try {
                const urls =
                  typeof data.urls === "string"
                    ? JSON.parse(data.urls)
                    : data.urls;

                const newSearchInfo = {
                  stages: searchData
                    ? [...searchData.stages, "reading"]
                    : ["reading"],
                  query: searchData?.query || "",
                  urls: urls,
                };
                searchData = newSearchInfo;

                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiResponseId
                      ? {
                          ...msg,
                          content: streamedContent,
                          searchInfo: newSearchInfo,
                          isLoading: false,
                        }
                      : msg
                  )
                );
              } catch (err) {
                console.error("Error parsing search results:", err);
              }
            } else if (data.type === "end") {
              console.log("✅ Stream ended");
              if (searchData) {
                const finalSearchInfo = {
                  ...searchData,
                  stages: [...searchData.stages, "writing"],
                };

                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiResponseId
                      ? {
                          ...msg,
                          searchInfo: finalSearchInfo,
                          isLoading: false,
                        }
                      : msg
                  )
                );
              } else {
                // Mark as not loading if no search data
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiResponseId ? { ...msg, isLoading: false } : msg
                  )
                );
              }

              eventSource.close();
            } else {
              console.log("⚠️ Unknown event type:", data.type, data);
            }
          } catch (parseError) {
            console.error("❌ JSON Parse error:", parseError);
            console.log("📝 Problematic data:", event.data);

            // Try to handle as raw text if JSON parsing fails
            if (event.data && typeof event.data === "string") {
              // Check if it looks like a data: line from SSE
              if (event.data.startsWith("data: ")) {
                const jsonPart = event.data.substring(6); // Remove 'data: ' prefix
                try {
                  const data = JSON.parse(jsonPart);
                  // Recursively call the same logic
                  console.log("✅ Successfully parsed after cleaning:", data);
                  // You could recursively process this, but for simplicity, just log it
                } catch (secondError) {
                  console.error(
                    "❌ Still can't parse after cleaning:",
                    secondError
                  );
                  console.log("📝 Cleaned data:", jsonPart);

                  // Fallback: treat as plain text content
                  if (jsonPart.trim()) {
                    streamedContent += jsonPart;
                    hasReceivedContent = true;

                    setMessages((prev) =>
                      prev.map((msg) =>
                        msg.id === aiResponseId
                          ? {
                              ...msg,
                              content: streamedContent,
                              isLoading: false,
                            }
                          : msg
                      )
                    );
                  }
                }
              } else {
                // Treat as plain text content
                streamedContent += event.data;
                hasReceivedContent = true;

                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === aiResponseId
                      ? { ...msg, content: streamedContent, isLoading: false }
                      : msg
                  )
                );
              }
            }
          }
        };

        // Handle errors
        eventSource.onerror = (error) => {
          console.error("❌ EventSource error:", error);
          console.log("EventSource readyState:", eventSource.readyState);
          eventSource.close();

          // Only update with error if we don't have content yet
          if (!streamedContent) {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === aiResponseId
                  ? {
                      ...msg,
                      content:
                        "Sorry, there was an error processing your request.",
                      isLoading: false,
                    }
                  : msg
              )
            );
          }
        };

        // Listen for end event
        eventSource.addEventListener("end", () => {
          eventSource.close();
        });
      } catch (error) {
        console.error("Error setting up EventSource:", error);
        setMessages((prev) => [
          ...prev,
          {
            id: newMessageId + 1,
            content: "Sorry, there was an error connecting to the server.",
            isUser: false,
            type: "message",
            isLoading: false,
          },
        ]);
      }
    }
  };

  return (
    <div className="flex justify-center bg-gray-100 min-h-screen py-8 px-4">
      {/* Main container with refined shadow and border */}
      <div className="w-[70%] bg-white flex flex-col rounded-xl shadow-lg border border-gray-100 overflow-hidden h-[90vh]">
        <Header />
        <MessageArea messages={messages} />
        <InputBar
          currentMessage={currentMessage}
          setCurrentMessage={setCurrentMessage}
          onSubmit={handleSubmit}
        />
      </div>
    </div>
  );
};

export default Home;
