"""
Database Module

SQLite-based storage for fabric catalog and search index.
Handles persistence of image features for fast retrieval during search.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
import pickle


class Database:
    """
    SQLite database for fabric catalog and search index.
    """
    
    def __init__(self, db_path: str = "data/fabric_search.db"):
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_db()
    
    def build_augmented_index(self):
        """
        Build index with augmented versions of catalog images
        This makes the index rotation-invariant
        """
        print("Building augmented search index...")
        
        for fabric in self.get_all_fabrics():
            img = cv2.imread(fabric['image_path'])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Original
            self._index_single_image(fabric, img_rgb, suffix='')
            
            # Rotations (for fabrics that might be photographed at different angles)
            for angle, suffix in [(90, '_r90'), (180, '_r180'), (270, '_r270')]:
                rotated = cv2.rotate(img_rgb, 
                                    cv2.ROTATE_90_CLOCKWISE if angle == 90
                                    else cv2.ROTATE_180 if angle == 180
                                    else cv2.ROTATE_90_COUNTERCLOCKWISE)
                self._index_single_image(fabric, rotated, suffix=suffix)
        
        print(f"Index built with {len(self.catalog_index)} variations")

    def _index_single_image(self, fabric: dict, img: np.ndarray, suffix: str):
        """
        Index a single image variation
        """
        features = self.image_processor.extract_features(img)
        
        # Store with suffix to link back to original design
        self.catalog_index.append({
            'design_no': fabric['design_no'],
            'variation': suffix,  # Track which variation this is
            'image_path': fabric['image_path'],
            **features,
            **fabric  # metadata
        })
    
    def _ensure_db_dir(self):
        """Ensure database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Fabric catalog table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fabric_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                design_no TEXT UNIQUE NOT NULL,
                source_pdf TEXT,
                width TEXT,
                stock INTEGER,
                gsm INTEGER,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Search index table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fabric_id INTEGER UNIQUE NOT NULL,
                phash TEXT,
                dhash TEXT,
                ahash TEXT,
                whash TEXT,
                color_histogram BLOB,
                orb_descriptors BLOB,
                keypoint_count INTEGER,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (fabric_id) REFERENCES fabric_catalog(id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for fast hash lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_phash ON search_index(phash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dhash ON search_index(dhash)')
        
        conn.commit()
        conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def clear_all(self):
        """Clear all data from database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM search_index')
        cursor.execute('DELETE FROM fabric_catalog')
        conn.commit()
        conn.close()
    
    def insert_fabric(self, fabric_data: Dict) -> int:
        """
        Insert a fabric entry into the catalog.
        
        Returns:
            fabric_id of inserted row
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO fabric_catalog 
            (design_no, source_pdf, width, stock, gsm, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            fabric_data.get('design_no'),
            fabric_data.get('source_pdf'),
            fabric_data.get('width'),
            fabric_data.get('stock', 0),
            fabric_data.get('gsm', 0),
            fabric_data.get('image_path')
        ))
        
        fabric_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return fabric_id
    
    def insert_index(self, fabric_id: int, features: Dict):
        """
        Insert search index for a fabric.
        
        Args:
            fabric_id: ID of the fabric in catalog
            features: Extracted features from ImageProcessor
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Serialize numpy arrays
        color_hist_blob = pickle.dumps(features['color_histogram'])
        orb_desc_blob = pickle.dumps(features['orb_descriptors']) if features['orb_descriptors'] is not None else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO search_index
            (fabric_id, phash, dhash, ahash, whash, color_histogram, orb_descriptors, keypoint_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            fabric_id,
            features['hashes']['phash'],
            features['hashes']['dhash'],
            features['hashes']['ahash'],
            features['hashes']['whash'],
            color_hist_blob,
            orb_desc_blob,
            features['keypoint_count']
        ))
        
        conn.commit()
        conn.close()
    
    def get_all_indexed_fabrics(self) -> List[Dict]:
        """
        Get all fabrics with their index data.
        
        Returns:
            List of fabric entries with features
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                fc.id, fc.design_no, fc.source_pdf, fc.width, fc.stock, fc.gsm, fc.image_path,
                si.phash, si.dhash, si.ahash, si.whash, 
                si.color_histogram, si.orb_descriptors, si.keypoint_count
            FROM fabric_catalog fc
            JOIN search_index si ON fc.id = si.fabric_id
        ''')
        
        results = []
        for row in cursor.fetchall():
            fabric = dict(row)
            
            # Deserialize numpy arrays
            if fabric['color_histogram']:
                fabric['color_histogram'] = pickle.loads(fabric['color_histogram'])
            if fabric['orb_descriptors']:
                fabric['orb_descriptors'] = pickle.loads(fabric['orb_descriptors'])
            
            results.append(fabric)
        
        conn.close()
        return results
    
    def get_fabric_by_design(self, design_no: str) -> Optional[Dict]:
        """Get fabric by design number."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM fabric_catalog WHERE design_no = ?
        ''', (design_no,))
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_index_count(self) -> int:
        """Get count of indexed fabrics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM search_index')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_catalog_count(self) -> int:
        """Get count of fabrics in catalog."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM fabric_catalog')
        count = cursor.fetchone()[0]
        conn.close()
        return count

class IndexManager:
    """
    Manages building and loading the search index.
    """
    
    def __init__(self, db: Database, image_processor):
        self.db = db
        self.processor = image_processor
        self._cached_index: Optional[List[Dict]] = None
    
    def build_index(self, catalog_json_path: str, images_base_path: str, 
                    rebuild: bool = False) -> Dict[str, Any]:
        """
        Build search index from catalog JSON.
        
        Args:
            catalog_json_path: Path to pdf_extracted_data.json
            images_base_path: Base path for images
            rebuild: If True, clear existing index first
            
        Returns:
            Stats about the indexing process
        """
        import json
        from pathlib import Path
        
        if rebuild:
            self.db.clear_all()
            self._cached_index = None
        
        # Load catalog
        with open(catalog_json_path, 'r') as f:
            catalog = json.load(f)
        
        stats = {
            'total': len(catalog),
            'indexed': 0,
            'failed': 0,
            'skipped': 0
        }
        
        for item in catalog:
            design_no = item.get('design_no')
            image_path = item.get('image_path')
            
            if not image_path:
                stats['skipped'] += 1
                continue
            
            # Build full image path
            full_image_path = str(Path(images_base_path) / image_path)
            
            # Check if already indexed
            existing = self.db.get_fabric_by_design(design_no)
            if existing and not rebuild:
                stats['skipped'] += 1
                continue
            
            try:
                # Extract features
                features = self.processor.extract_all_features(full_image_path)
                
                if features is None:
                    print(f"  Failed to extract features for {design_no}")
                    stats['failed'] += 1
                    continue
                
                # Insert into database
                fabric_id = self.db.insert_fabric(item)
                self.db.insert_index(fabric_id, features)
                
                stats['indexed'] += 1
                
                if stats['indexed'] % 50 == 0:
                    print(f"  Indexed {stats['indexed']} fabrics...")
                    
            except Exception as e:
                print(f"  Error indexing {design_no}: {e}")
                stats['failed'] += 1
        
        # Clear cache to force reload
        self._cached_index = None
        
        return stats
    
    def load_index(self) -> List[Dict]:
        """
        Load index into memory for fast search.
        Uses caching for subsequent calls.
        """
        if self._cached_index is None:
            self._cached_index = self.db.get_all_indexed_fabrics()
        return self._cached_index
    
    def get_index_stats(self) -> Dict[str, int]:
        """Get index statistics."""
        return {
            'catalog_count': self.db.get_catalog_count(),
            'index_count': self.db.get_index_count()
        }
