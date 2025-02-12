package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/go-chi/chi"
	"github.com/go-chi/chi/middleware"
	//"github.com/go-chi/chi/v5"
	//"github.com/go-chi/chi/v5/middleware"
	// "go.opentelemetry.io/otel"
	// "go.opentelemetry.io/otel/sdk/resource"
	// "go.opentelemetry.io/otel/sdk/trace"
	// "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	// "google.golang.org/grpc"
	// "go.opentelemetry.io/otel/trace"
)

// AppConfig holds configuration settings for the application
type AppConfig struct {
	vllm1_client *http.Client
	vllm2_client *http.Client
}

// Declare a global variable of AppConfig
var appConfig AppConfig

// Base URLs for the two vLLM processes (set to the root of the API)
const VLLM_1_BASE_URL = "http://localhost:8000/v1"
const VLLM_2_BASE_URL = "http://localhost:8001/v1"

/*
	Send a request to a vLLM process using a persistent client and return the response.

	Args:
	   client (httpx.AsyncClient): The persistent HTTPX client.
	   req_data (dict): The JSON payload to send.

Returns:

	httpx.Response: The response from the vLLM service.
*/
func send_request_to_vllm(client *http.Client,
	req_data map[string]interface{},
	ch chan<- string,
	errCh chan<- error) {

	jsonData, err := json.Marshal(req_data)
	if err != nil {
		fmt.Errorf("error marshalling JSON: %w", err)
		errCh <- err
		return
	}

	req, err := http.NewRequest("POST", "/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Errorf("error creating request: %w", err)
		errCh <- err
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		errCh <- err
		return
	}
	ch <- string(body)
}

func convertToJSON(reader io.Reader) (map[string]interface{}, error) {
	// Read the request body
	body, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, err
	}

	// Convert the body to a map (for simplicity, assuming key-value pairs)
	var jsonData map[string]interface{}
	err = json.Unmarshal(body, &jsonData)
	if err != nil {
		return nil, err
	}
	return jsonData, nil
}

func proxy_request(w http.ResponseWriter, req *http.Request) {
	// Extract the incoming request JSON data
	req_data, err := convertToJSON(req.Body)
	if err != nil {
		http.Error(w, "Error decoding JSON", http.StatusBadRequest)
		return
	}

	// Create channels to handle response and errors
	respCh := make(chan string)
	errCh := make(chan error)
	// Send request to vLLM-1 using the persistent client
	go send_request_to_vllm(appConfig.vllm1_client, req_data, respCh, errCh)

	fmt.Println("Doing other work asynchronously...")
	// Wait for the response or error
	select {
	case response1 := <-respCh:
		fmt.Println("Response1 received:")
		req2, err := http.NewRequest("POST", "/completions", strings.NewReader(response1))
		if err != nil {
			fmt.Errorf("error creating request: %w", err)
			return
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := appConfig.vllm2_client.Do(req2)
		if err != nil {
			fmt.Println("Error making request:", err)
			return
		}
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			w.Write([]byte(line))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}

			if err := scanner.Err(); err != nil {
				fmt.Fprintf(w, "error reading response: %v\n", err)
			}
		}
	case err := <-errCh:
		fmt.Println("Error occurred:")
		fmt.Println(err)
	}
}

func main() {
	// Create a channel to listen for OS signals
	signalChan := make(chan os.Signal, 1)
	done := make(chan bool, 1)

	// Register for interrupt and terminate signals
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)

	// Goroutine to handle shutdown signal
	go func() {
		sig := <-signalChan
		fmt.Println()
		fmt.Println(sig)
		// Perform shutdown tasks
		fmt.Println("Application shutting down...")
		time.Sleep(1 * time.Second) // Simulate cleanup tasks
		done <- true
	}()

	// Create and configure the HTTP client
	appConfig.vllm1_client = &http.Client{Timeout: 10 * time.Second}
	appConfig.vllm2_client = &http.Client{Timeout: 10 * time.Second}

	/*
	   // Create OTLP exporter
	   ctx := context.Background()
	   oltp_exporter, err := otlptracegrpc.New(ctx,
	       otlptracegrpc.WithInsecure(),
	       otlptracegrpc.WithEndpoint("localhost:4317"))

	   if err != nil {
	       return nil, fmt.Errorf("failed to create OTLP exporter: %w", err)
	   }

	   // Create a resource to identify this application
	   res, err := resource.New(ctx, resource.WithAttributes( semconv.ServiceNameKey.String("my-service"), ), )
	   if err != nil {
	       return nil, fmt.Errorf("failed to create resource: %w", err)
	   }

	   // Create TracerProvider with BatchSpanProcessor
	   tp := trace.NewTracerProvider( trace.WithBatcher(exporter), trace.WithResource(res), )

	   // Set the TracerProvider as the global provider
	   otel.SetTracerProvider(tp)
	*/
	r := chi.NewRouter()
	// Middleware
	r.Use(middleware.Logger)

	// Routes
	r.Post("/v1/completions", proxy_request)

	// Start listening for requests
	http.ListenAndServe(":8080", r)

	// Block until a shutdown signal is received
	fmt.Println("Application running... Press Ctrl+C to exit.")
	<-done
	fmt.Println("Application stopped.")
}
