# Star_Tracker_Code


Collection of files for creating a Tensorflow model that can be used for star identification.

The goal is to create a model that can identify a large number (>7000) of stars that can be used for the "lost-in-space" scenario.

Current Issues:
- Difficult to create a model that can reach high classification accuracy with only a few stars as inputs. (Current commercial star trackers use triangle algorithms that can identify with only 3 stars)
- High accuracy can be achieved with 15+ stars in the input, but this is problematic and perhaps less useful commercially.
