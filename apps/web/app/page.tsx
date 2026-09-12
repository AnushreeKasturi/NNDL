"use client";

import { FormEvent, useState } from "react";
import { apiFetch, LoginResponse } from "../lib/api";

type Contract = { id: string; title: string; text: string; created_at: string };
type Job = {
  id: string;
  contract_id: string;
  model_name: string;
  status: string;
  result_json?: Record<string, unknown>;
  error_message?: string;
};

export default function Home() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [token, setToken] = useState("");
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [modelName, setModelName] = useState("nlpaueb/legal-bert-base-uncased");
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [lastResult, setLastResult] = useState<string>("");
  const [loading, setLoading] = useState(false);

  async function register(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    try {
      await apiFetch("/auth/register", "POST", { email, password });
      setLastResult("Registered. You can now login.");
    } catch (err) {
      setLastResult(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function login(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await apiFetch<LoginResponse>("/auth/login", "POST", { email, password });
      setToken(response.access_token);
      setLastResult("Logged in.");
    } catch (err) {
      setLastResult(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function createContract(e: FormEvent) {
    e.preventDefault();
    if (!token) return;
    setLoading(true);
    try {
      const contract = await apiFetch<Contract>("/contracts", "POST", { title, text }, token);
      setContracts([contract, ...contracts]);
      setLastResult(`Contract created: ${contract.id}`);
    } catch (err) {
      setLastResult(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function refreshContracts() {
    if (!token) return;
    setLoading(true);
    try {
      const rows = await apiFetch<Contract[]>("/contracts", "GET", undefined, token);
      setContracts(rows);
    } catch (err) {
      setLastResult(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function analyze(contractId: string) {
    if (!token) return;
    setLoading(true);
    try {
      const job = await apiFetch<Job>(
        `/contracts/${contractId}/analyze`,
        "POST",
        { model_name: modelName },
        token
      );
      setJobs([job, ...jobs]);
      setLastResult(`Job queued: ${job.id}`);
    } catch (err) {
      setLastResult(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function refreshJobs() {
    if (!token) return;
    setLoading(true);
    try {
      const rows = await apiFetch<Job[]>("/jobs", "GET", undefined, token);
      setJobs(rows);
      const firstDone = rows.find((job) => job.status === "succeeded" && job.result_json);
      if (firstDone) {
        setLastResult(JSON.stringify(firstDone.result_json, null, 2));
      }
    } catch (err) {
      setLastResult(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Legal Risk Classifier</h1>
      <p>Production-ready starter: auth, contract intake, async inference jobs, result tracking.</p>

      <section>
        <h2>Auth</h2>
        <form className="row" onSubmit={register}>
          <div>
            <label>Email</label>
            <input value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>
          <div>
            <label>Password</label>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
          </div>
          <button disabled={loading}>Register</button>
          <button disabled={loading} onClick={login}>
            Login
          </button>
        </form>
        <small>{token ? "Authenticated" : "Not authenticated"}</small>
      </section>

      <section>
        <h2>Create Contract</h2>
        <form onSubmit={createContract}>
          <label>Title</label>
          <input value={title} onChange={(e) => setTitle(e.target.value)} />
          <label>Contract Text</label>
          <textarea rows={8} value={text} onChange={(e) => setText(e.target.value)} />
          <button disabled={loading || !token}>Create Contract</button>
        </form>
      </section>

      <section>
        <h2>Contracts</h2>
        <button onClick={refreshContracts} disabled={loading || !token}>
          Refresh Contracts
        </button>
        <label>Model Name</label>
        <input value={modelName} onChange={(e) => setModelName(e.target.value)} />
        {contracts.map((contract) => (
          <div key={contract.id}>
            <strong>{contract.title}</strong>
            <button disabled={loading} onClick={() => analyze(contract.id)}>
              Analyze
            </button>
          </div>
        ))}
      </section>

      <section>
        <h2>Jobs</h2>
        <button onClick={refreshJobs} disabled={loading || !token}>
          Refresh Jobs
        </button>
        {jobs.map((job) => (
          <div key={job.id}>
            <strong>{job.id}</strong> - {job.status} ({job.model_name})
          </div>
        ))}
      </section>

      <section>
        <h2>Latest Result</h2>
        <pre>{lastResult}</pre>
      </section>
    </main>
  );
}

