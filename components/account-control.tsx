"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { LockIcon } from "@/components/icons";
import { authClient } from "@/lib/auth-client";

export function AccountControl({ userName }: { userName: string }) {
  const router = useRouter();
  const [isPending, setIsPending] = useState(false);

  async function signOut() {
    setIsPending(true);
    await authClient.signOut({
      fetchOptions: {
        onSuccess: () => {
          router.push("/login");
          router.refresh();
        },
        onError: () => setIsPending(false),
      },
    });
  }

  return (
    <div className="account-control">
      <span className="private-badge" title={`Logget ind som ${userName}`}>
        <LockIcon /> {userName}
      </span>
      <button type="button" onClick={signOut} disabled={isPending}>
        {isPending ? "Logger ud …" : "Log ud"}
      </button>
    </div>
  );
}
