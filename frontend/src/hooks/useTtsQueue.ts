import { useCallback, useEffect, useRef, useState } from "react";
import { synthesizeTTSStream } from "../api";

type QueueItem = { text: string; url: string };

type TtsQueueState = {
  enqueue: (text: string) => void;
  queueSize: number;
  playing: boolean;
  generating: boolean;
  error: string | null;
  clearError: () => void;
};

const MAX_CACHE_ITEMS = 30;

export function useTtsQueue(): TtsQueueState {
  const queueRef = useRef<QueueItem[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const cacheRef = useRef<Map<string, string>>(new Map());

  const [queueSize, setQueueSize] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const syncQueueSize = useCallback(() => {
    setQueueSize(queueRef.current.length);
  }, []);

  const playNext = useCallback(() => {
    if (!audioRef.current) {
      audioRef.current = new Audio();
      audioRef.current.onended = () => {
        setPlaying(false);
        playNext();
      };
      audioRef.current.onerror = () => {
        setPlaying(false);
        setError("TTS playback failed.");
        playNext();
      };
    }

    const next = queueRef.current.shift();
    syncQueueSize();
    if (!next) return;

    setPlaying(true);
    audioRef.current.src = next.url;
    void audioRef.current.play().catch(() => {
      setPlaying(false);
      setError("TTS playback failed.");
      playNext();
    });
  }, [syncQueueSize]);

  const enqueue = useCallback(
    (text: string) => {
      const normalized = text.trim();
      if (!normalized) return;

      const cachedUrl = cacheRef.current.get(normalized);
      if (cachedUrl) {
        queueRef.current.push({ text: normalized, url: cachedUrl });
        syncQueueSize();
        if (!playing) playNext();
        return;
      }

      setGenerating(true);
      setError(null);
      synthesizeTTSStream(normalized)
        .then(({ url, done }) => {
          queueRef.current.push({ text: normalized, url });
          syncQueueSize();
          if (!playing) playNext();

          done
            .then(() => {
              cacheRef.current.set(normalized, url);
              if (cacheRef.current.size > MAX_CACHE_ITEMS) {
                const oldestKey = cacheRef.current.keys().next().value as string | undefined;
                if (oldestKey) {
                  const oldestUrl = cacheRef.current.get(oldestKey);
                  if (oldestUrl) URL.revokeObjectURL(oldestUrl);
                  cacheRef.current.delete(oldestKey);
                }
              }
            })
            .catch((e: unknown) => {
              setError(e instanceof Error ? e.message : "TTS failed.");
            });
        })
        .catch((e: unknown) => {
          setError(e instanceof Error ? e.message : "TTS failed.");
        })
        .finally(() => {
          setGenerating(false);
        });
    },
    [playNext, playing, syncQueueSize]
  );

  const clearError = useCallback(() => setError(null), []);

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.onended = null;
        audioRef.current.onerror = null;
        audioRef.current.pause();
        audioRef.current.src = "";
        audioRef.current = null;
      }
      for (const url of cacheRef.current.values()) {
        URL.revokeObjectURL(url);
      }
      cacheRef.current.clear();
      queueRef.current = [];
    };
  }, []);

  return { enqueue, queueSize, playing, generating, error, clearError };
}
