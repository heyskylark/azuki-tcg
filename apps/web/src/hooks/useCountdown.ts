"use client";

import { useState, useEffect } from "react";

interface UseCountdownReturn {
  secondsRemaining: number;
  isExpired: boolean;
}

export function useCountdown(deadline: string | null): UseCountdownReturn {
  const [secondsRemaining, setSecondsRemaining] = useState(0);
  const [isExpired, setIsExpired] = useState(false);

  useEffect(() => {
    if (!deadline) {
      setSecondsRemaining(0);
      setIsExpired(false);
      return;
    }

    const calculateRemaining = () => {
      const deadlineTime = new Date(deadline).getTime();
      const now = Date.now();
      const remaining = Math.max(0, Math.ceil((deadlineTime - now) / 1000));
      return remaining;
    };

    // Initial calculation
    const initial = calculateRemaining();
    setSecondsRemaining(initial);
    setIsExpired(initial <= 0);

    if (initial <= 0) {
      return;
    }

    // Update every second
    const interval = setInterval(() => {
      const remaining = calculateRemaining();
      setSecondsRemaining(remaining);

      if (remaining <= 0) {
        setIsExpired(true);
        clearInterval(interval);
      }
    }, 1000);

    return () => {
      clearInterval(interval);
    };
  }, [deadline]);

  return { secondsRemaining, isExpired };
}
