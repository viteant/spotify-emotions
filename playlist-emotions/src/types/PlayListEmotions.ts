export type NormalizeEmotion = {
  normalize: string;
  sum_avg: number;
  emotion_count: number;
}

export type EmotionScores = Record<string, number>;

export interface EmotionDictionaryEntry {
  emotion: string;  // nombre de la emoción
  emoji: string;    // emoji asociado
}
