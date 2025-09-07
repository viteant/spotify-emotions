export type SortDir = 'asc' | 'desc'

export type SortBy =
  | 'track'
  | 'artist'
  | 'album'
  | 'full_positive'
  | 'positive'
  | 'neutral'
  | 'negative'
  | 'full_negative'

export interface TrackItem {
  track: string
  artist: string
  album: string
  full_positive: number
  positive: number
  neutral: number
  negative: number
  full_negative: number
  emotions: string | EmotionEntry[]
}

export interface TracksResponse {
  page: number
  page_size: number
  total: number
  sort_by: SortBy
  sort_dir: SortDir
  items: TrackItem[]
}

export interface EmotionEntry {
  label: string
  score: number
}

export interface Emotion {
  emotion: string
  normalize: string
  emoji: string
}

export type Filters = { artist: string; track: string; album: string }
