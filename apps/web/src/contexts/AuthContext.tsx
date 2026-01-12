"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";
import { useRouter } from "next/navigation";
import {
  getIdentityUser,
  type IdentityUser,
} from "@/lib/auth/identity";

interface User {
  id: string;
  username: string;
  displayName: string;
  email: string;
}

interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (
    username: string,
    email: string,
    password: string
  ) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
  checkAuth: () => boolean;
}

const AuthContext = createContext<AuthContextValue | null>(null);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  const identityToUser = (identity: IdentityUser): User => ({
    id: identity.userId,
    username: identity.username,
    displayName: identity.displayName,
    email: identity.email,
  });

  const checkAuth = useCallback((): boolean => {
    const identity = getIdentityUser();
    if (identity) {
      setUser(identityToUser(identity));
      return true;
    }
    setUser(null);
    return false;
  }, []);

  useEffect(() => {
    checkAuth();
    setIsLoading(false);
  }, [checkAuth]);

  const login = async (email: string, password: string): Promise<void> => {
    const response = await fetch("/api/users/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || "Login failed");
    }

    const data = await response.json();
    setUser({
      id: data.user.id,
      username: data.user.username,
      displayName: data.user.displayName,
      email: data.user.email,
    });

    router.push("/dashboard");
  };

  const signup = async (
    username: string,
    email: string,
    password: string
  ): Promise<void> => {
    const response = await fetch("/api/users/auth/sign-up", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || "Signup failed");
    }

    const data = await response.json();
    setUser({
      id: data.user.id,
      username: data.user.username,
      displayName: data.user.displayName,
      email: data.user.email,
    });

    router.push("/dashboard");
  };

  const logout = async (): Promise<void> => {
    try {
      await fetch("/api/users/auth/logout", {
        method: "POST",
      });
    } catch {
      // Ignore errors, still clear local state
    }

    setUser(null);
    router.push("/login");
  };

  const refresh = async (): Promise<void> => {
    const response = await fetch("/api/users/auth/refresh", {
      method: "POST",
    });

    if (!response.ok) {
      setUser(null);
      throw new Error("Token refresh failed");
    }

    const data = await response.json();
    setUser({
      id: data.user.id,
      username: data.user.username,
      displayName: data.user.displayName,
      email: data.user.email,
    });
  };

  const value: AuthContextValue = {
    user,
    isAuthenticated: !!user,
    isLoading,
    login,
    signup,
    logout,
    refresh,
    checkAuth,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
