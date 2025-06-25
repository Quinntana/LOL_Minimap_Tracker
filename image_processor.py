"""Screen capture and image processing for League of Legends champion tracking."""
import mss
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from skimage.metrics import structural_similarity as ssim
from utils import setup_logger, timeit

class ImageProcessor:
    """Handles screen capture and image processing for champion detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image processor with configuration.
        
        Args:
            config: Configuration dictionary containing image processing settings
        """
        self.logger = setup_logger("image_processor")
        self.config = config
        
        # Color detection settings
        self.lower_red = np.array([100, 0, 0])
        self.upper_red = np.array([255, 100, 100])
        
        self.lower_white = np.array([200, 200, 200])
        self.upper_white = np.array([255, 255, 255])
        
        # Circle detection settings
        self.circle_radius_min = config.get("circle_radius_min", 12)
        self.circle_radius_max = config.get("circle_radius_max", 40)
        
        # SSIM threshold for champion recognition
        self.ssim_threshold = config.get("ssim_threshold", 0.3)
    
    def capture_screen(self) -> np.ndarray:
        """
        Capture the screen region specified in config.
        
        Returns:
            Image as numpy array in BGR format
        """
        monitor = {
            "top": self.config.get("top", 850),
            "left": self.config.get("left", 1500),
            "width": self.config.get("width", 240),
            "height": self.config.get("height", 240)
        }
        
        with mss.mss() as sct:
            screen_img = sct.grab(monitor)
            img = np.array(screen_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            return img
    
    def detect_red_circles(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect red circles in the image.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            Array of detected circles or None if none found
        """
        # Extract red regions
        mask_red = cv2.inRange(img, self.lower_red, self.upper_red)
        
        # Apply blur to reduce noise
        mask_red_blurred = cv2.GaussianBlur(mask_red, (3, 3), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            mask_red_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=self.circle_radius_min,
            maxRadius=self.circle_radius_max
        )
        
        return circles
    
    def find_rectangle_center(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find center of white rectangle in the image.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            Tuple of (x, y) coordinates or None if not found
        """
        # Extract white regions
        mask_white = cv2.inRange(img, self.lower_white, self.upper_white)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask_white = cv2.dilate(mask_white, kernel, iterations=2)
        mask_white = cv2.erode(mask_white, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find rectangle that matches size criteria
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 60 < w < 160 and 20 < h < 100:
                return (x + w // 2, y + h // 2)
        
        return None
    
    @staticmethod
    def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate structural similarity between two images.
        
        Args:
            img1: First image in BGR format
            img2: Second image in BGR format
            
        Returns:
            SSIM score between 0.0 and 1.0
        """
        try:
            # Convert to grayscale
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Calculate SSIM
            score, _ = ssim(img1_gray, img2_gray, full=True)
            return float(score)
        except Exception as e:
            return 0.0
    
    def find_best_match(
        self, 
        query_img: np.ndarray, 
        reference_images: Dict[str, np.ndarray]
    ) -> Tuple[Optional[str], float]:
        """
        Find best matching champion for the query image.
        
        Args:
            query_img: Image to match in BGR format
            reference_images: Dictionary mapping champion names to reference images
            
        Returns:
            Tuple of (champion_name, similarity_score) or (None, 0.0) if no match
        """
        best_match = None
        highest_score = self.ssim_threshold
        
        for name, ref_img in reference_images.items():
            try:
                # Resize reference image to match query image
                resized_img = cv2.resize(ref_img, (query_img.shape[1], query_img.shape[0]))
                
                # Calculate similarity
                score = self.calculate_ssim(query_img, resized_img)
                
                # Update best match if score is higher
                if score > highest_score:
                    highest_score = score
                    best_match = name
            except Exception as e:
                self.logger.warning(f"Error matching {name}: {e}")
        
        return best_match, highest_score
    
    @timeit
    def detect_champions(
        self, 
        img: np.ndarray, 
        reference_images: Dict[str, np.ndarray]
    ) -> List[Tuple[str, int, int, float]]:
        """
        Detect champions in the image.
        
        Args:
            img: Input image in BGR format
            reference_images: Dictionary mapping champion names to reference images
            
        Returns:
            List of tuples (champion_name, x, y, similarity_score)
        """
        # Detect red circles
        circles = self.detect_red_circles(img)
        
        if circles is None or img.size == 0:
            return []
        
        detected = []
        
        # Process each circle
        for (x, y, r) in np.round(circles[0, :]).astype(int):
            # Extract region around circle
            y1, y2 = max(0, y-r), min(img.shape[0], y+r)
            x1, x2 = max(0, x-r), min(img.shape[1], x+r)
            
            # Skip if region is invalid
            if y1 >= y2 or x1 >= x2:
                continue
            
            # Extract region
            region = img[y1:y2, x1:x2]
            
            # Skip if region is empty
            if region.size == 0:
                continue
            
            # Find best match
            champion, score = self.find_best_match(region, reference_images)
            
            # Add to results if match found
            if champion:
                detected.append((champion, x, y, score))
        
        return detected