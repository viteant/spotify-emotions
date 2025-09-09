export interface Track {
  track_spotify_id: string
  cluster_id: number
  track_name: string
  cluster_name: string
  artist_spotify_id: string
  artist_name: string
  album_spotify_id: string
  album_name: string
}

export interface Playlist {
  title: string
  description?: string
  cluster_id: number
  tracks: Track[]
}

export interface Cluster {
  id: number
  name: string
  description: string
}

export interface ClusterPlaylistBody {
  description?: string
  public?: boolean
  replace?: boolean
}

export interface ClusterPlaylistResponse {
  cluster_id: number
  cluster_name: string
  tracks_count: number
  playlist_id: string
  playlist_url: string
  mode: 'replace' | 'append'
}
