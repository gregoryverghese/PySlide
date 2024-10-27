import os
import pickle
import numpy as np

class DiskWrite:
    def __init__(self, path, write_frequency=10):
        self.path = path
        self.write_frequency = write_frequency
        os.makedirs(path, exist_ok=True)

    def __repr__(self):
        return f'DiskWrite(path: {self.path})'

    def _print_progress(self, i):
        print(f"Processed {i + 1} tiles", end='\r')

    def write(self, parser):
        """
        Writes tiles to disk in batches after every self.write_frequency iterations.

        param: parser Generator: A generator that yields (coordinates, tile) tuples.
        """
        tile_buffer = []
        meta_buffer = []

        for i, (p, tile) in enumerate(parser):
            # Create file names for tile and metadata
            name = str(p[1]) + '_' + str(p[0])
            tile_path = os.path.join(self.path, f"{name}.npy")
            meta_path = os.path.join(self.path, f"{name}_meta.pkl")

            # Accumulate tile and metadata in buffers
            tile_buffer.append((tile_path, tile))
            meta_buffer.append((meta_path, {'size': tile.shape, 'dtype': tile.dtype}))

            # Write to disk if the buffer reaches the specified frequency
            if (i + 1) % self.write_frequency == 0:
                self._write_buffer(tile_buffer, meta_buffer)
                tile_buffer = []  # Clear the buffer after writing
                meta_buffer = []  # Clear the metadata buffer
                #self._print_progress(i)

        # Write any remaining tiles in the buffer after the loop completes
        if tile_buffer:
            self._write_buffer(tile_buffer, meta_buffer)

        print("\nFinished writing tiles to disk.")

    def _write_buffer(self, tile_buffer, meta_buffer):
        """
        Writes the contents of the buffers to disk.
        
        param: tile_buffer List[Tuple[str, np.ndarray]]: A list of tuples (file path, tile).
        param: meta_buffer List[Tuple[str, dict]]: A list of tuples (file path, metadata).
        """
        # Write all tiles in the buffer to disk
        for tile_path, tile in tile_buffer:
            np.save(tile_path, tile)

        # Write corresponding metadata to disk
        for meta_path, metadata in meta_buffer:
            with open(meta_path, 'wb') as meta_file:
                pickle.dump(metadata, meta_file)

