"use client";

import { useState } from "react";

import { authClient } from "@/lib/auth-client";

function GitHubMark() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path
        fill="currentColor"
        d="M12 .7a11.5 11.5 0 0 0-3.64 22.4c.58.1.79-.25.79-.56v-2.2c-3.22.7-3.9-1.37-3.9-1.37-.52-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.72 1.27 3.38.97.1-.75.4-1.27.74-1.56-2.57-.29-5.28-1.28-5.28-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.2-1.49 3.16-1.18 3.16-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.41-2.71 5.38-5.29 5.67.42.36.79 1.07.79 2.16v3.2c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z"
      />
    </svg>
  );
}

export function LoginCard({ rejected }: { rejected: boolean }) {
  const [isPending, setIsPending] = useState(false);
  const [clientError, setClientError] = useState<string | null>(null);

  async function signIn() {
    setIsPending(true);
    setClientError(null);

    try {
      const result = await authClient.signIn.social({
        provider: "github",
        callbackURL: "/",
        errorCallbackURL: "/login?error=access_denied",
      });

      if (!result.error) return;
      setClientError("Login kunne ikke startes. Prøv igen om et øjeblik.");
    } catch {
      setClientError("Login kunne ikke startes. Prøv igen om et øjeblik.");
    }
    setIsPending(false);
  }

  return (
    <main className="login-shell">
      <section className="login-card" aria-labelledby="login-heading">
        <div className="login-brand" aria-hidden="true">F</div>
        <p className="eyebrow eyebrow--lime">Privat beslutningsværktøj</p>
        <h1 id="login-heading">Log ind på FPL HoldPlanner</h1>
        <p className="login-intro">
          Adgangen er låst til din GitHub-konto. Der gemmes ingen separat adgangskode i appen.
        </p>

        {(rejected || clientError) && (
          <p className="login-error" role="alert">
            {clientError ?? "GitHub-kontoen har ikke adgang, eller login blev afbrudt."}
          </p>
        )}

        <button className="github-login" type="button" onClick={signIn} disabled={isPending}>
          <GitHubMark />
          {isPending ? "Sender dig til GitHub …" : "Fortsæt med GitHub"}
        </button>

        <div className="login-security">
          <span aria-hidden="true">✓</span>
          <p><strong>Kun din konto.</strong> Din numeriske GitHub-identitet kontrolleres ved hvert nyt login.</p>
        </div>
      </section>
    </main>
  );
}
