'use client'

import { useMemo, useState, useEffect, type ReactNode } from 'react'
import useSWR from 'swr'
import { AnimatePresence, motion } from 'motion/react'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BrainCircuit,
  Database,
  Network,
  RefreshCw,
  Search,
  ShieldCheck,
  Target,
  Zap,
} from 'lucide-react'
import { fetchBackend, numberValue, records, type ApiRecord } from '@/lib/backend-api'

const cyan = '#22d3ee'
const violet = '#a855f7'
const coral = '#f43f5e'
const amber = '#fbbf24'
const emerald = '#34d399'

type ViewId = 'overview' | 'explorer' | 'models' | 'network' | 'inference'
type Feed = { mutate: () => void }

const nav: Array<{ id: ViewId; label: string; icon: typeof Activity; index: string }> = [
  { id: 'overview', label: 'Overview', icon: Activity, index: '01' },
  { id: 'explorer', label: 'Explorer', icon: Target, index: '02' },
  { id: 'models', label: 'Model Lab', icon: BrainCircuit, index: '03' },
  { id: 'network', label: 'Network', icon: Network, index: '04' },
  { id: 'inference', label: 'Inference', icon: Zap, index: '05' },
]

function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`glass-card ${className}`}>{children}</section>
}

function Label({ children }: { children: ReactNode }) {
  return <span className="eyebrow">{children}</span>
}

function pick(row: ApiRecord | undefined, keys: string[]) {
  if (!row) return undefined
  const key = keys.find((candidate) => row[candidate] !== undefined)
  return key ? row[key] : undefined
}

function text(row: ApiRecord | undefined, keys: string[], fallback = '—') {
  if (!row) return fallback
  const value = pick(row, keys)
  return value === undefined || value === null || value === '' ? fallback : String(value)
}

function num(row: ApiRecord | undefined, keys: string[], decimals = 4, fallback = '—'): string {
  if (!row) return fallback
  const val = numberValue(row, keys)
  if (val === null || val === undefined) return fallback
  return val.toFixed(decimals)
}

function formatInt(value: unknown, fallback = '—'): string {
  if (value === undefined || value === null) return fallback
  const parsed = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(parsed) ? parsed.toLocaleString() : fallback
}

function Empty({ message = 'No records were returned by this endpoint.' }: { message?: string }) {
  return (
    <div className="empty-state">
      <Database size={24} />
      <span>{message}</span>
    </div>
  )
}

function Loading() {
  return (
    <div className="loading-grid">
      <div />
      <div />
      <div />
    </div>
  )
}

function ErrorState({ error, retry }: { error: Error; retry: () => void }) {
  return (
    <div className="error-state">
      <AlertTriangle size={18} />
      <span>{error.message}</span>
      <button className="sync-btn" onClick={retry}>
        Retry
      </button>
    </div>
  )
}

function SectionHead({ eyebrow, title, meta }: { eyebrow: string; title: string; meta?: string }) {
  return (
    <div className="section-head">
      <div>
        <Label>{eyebrow}</Label>
        <h2>{title}</h2>
      </div>
      {meta && <span className="stage-live">{meta}</span>}
    </div>
  )
}

/* =========================================================================
   One Large Elegant Transaction-Flow Visualization (Illustrative Schematic)
   - Clean, purposeful nodes indicating topological roles
   - Smooth directional paths with arrow corridors
   - Clearly marked illustrative schematic; dynamic top entity linked from data
   - Prominent 13px legend
   ========================================================================= */
function FlowField({ topEntityId }: { topEntityId?: string }) {
  const targetLabel = topEntityId ? `Tx ${topEntityId} [Target Hub]` : 'Flagged Entity Hub'

  return (
    <div className="flow-field">
      <div className="flow-field-head">
        <div className="flow-head-left">
          <i className="live-dot" />
          <span>Illustrative Flow Schematic · Directed Edge Topology</span>
        </div>
        <div className="flow-head-right">
          <span>SOURCE → AGGREGATION → HUB → EGRESS</span>
        </div>
      </div>

      <div className="flow-canvas">
        <svg
          viewBox="0 0 960 380"
          role="img"
          aria-label="Illustrative directional transaction flow schematic showing Bitcoin communities and risk propagation"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <linearGradient id="flow-grad-cyan" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.85" />
              <stop offset="100%" stopColor="#a855f7" stopOpacity="0.85" />
            </linearGradient>
            <linearGradient id="flow-grad-risk" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#fbbf24" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#f43f5e" stopOpacity="0.95" />
            </linearGradient>

            <filter id="subtle-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2.2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>

            <marker id="arrow-cyan" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="3.5" markerHeight="3.5" orient="auto">
              <path d="M 0 1 L 7 4 L 0 7 z" fill="#22d3ee" fillOpacity="0.85" />
            </marker>
            <marker id="arrow-violet" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="3.5" markerHeight="3.5" orient="auto">
              <path d="M 0 1 L 7 4 L 0 7 z" fill="#a855f7" fillOpacity="0.85" />
            </marker>
            <marker id="arrow-risk" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="3.8" markerHeight="3.8" orient="auto">
              <path d="M 0 1 L 7 4 L 0 7 z" fill="#f43f5e" fillOpacity="0.95" />
            </marker>
            <marker id="arrow-slate" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="3.5" markerHeight="3.5" orient="auto">
              <path d="M 0 1 L 7 4 L 0 7 z" fill="#94a3b8" fillOpacity="0.6" />
            </marker>
          </defs>

          {/* 1. Community Regions */}
          <g className="community-regions">
            <rect x="25" y="30" width="185" height="320" rx="14" fill="rgba(34,211,238,0.03)" stroke="rgba(34,211,238,0.2)" strokeDasharray="5 5" strokeWidth="1.2" />
            <text x="40" y="54" fill="rgba(34,211,238,0.85)" fontFamily="var(--font-mono, monospace)" fontSize="11" fontWeight="700" letterSpacing="0.08em">
              COMMUNITY 01 · INFLOW
            </text>

            <rect x="245" y="30" width="225" height="320" rx="14" fill="rgba(168,85,247,0.03)" stroke="rgba(168,85,247,0.2)" strokeDasharray="5 5" strokeWidth="1.2" />
            <text x="260" y="54" fill="rgba(168,85,247,0.85)" fontFamily="var(--font-mono, monospace)" fontSize="11" fontWeight="700" letterSpacing="0.08em">
              COMMUNITY 14 · AGGREGATION
            </text>

            <rect x="505" y="30" width="225" height="320" rx="14" fill="rgba(244,63,94,0.035)" stroke="rgba(244,63,94,0.25)" strokeDasharray="5 5" strokeWidth="1.2" />
            <text x="520" y="54" fill="rgba(244,63,94,0.9)" fontFamily="var(--font-mono, monospace)" fontSize="11" fontWeight="700" letterSpacing="0.08em">
              COMMUNITY 28 · SUSPICIOUS HUB
            </text>

            <rect x="765" y="30" width="170" height="320" rx="14" fill="rgba(34,211,238,0.03)" stroke="rgba(34,211,238,0.2)" strokeDasharray="5 5" strokeWidth="1.2" />
            <text x="780" y="54" fill="rgba(34,211,238,0.85)" fontFamily="var(--font-mono, monospace)" fontSize="11" fontWeight="700" letterSpacing="0.08em">
              SETTLEMENT &amp; EGRESS
            </text>
          </g>

          {/* 2. Directional Transfer Paths */}
          <g className="flow-paths">
            <path d="M 130 115 C 195 115, 240 135, 335 135" fill="none" stroke="rgba(34,211,238,0.5)" strokeWidth="1.8" markerEnd="url(#arrow-cyan)" />
            <path d="M 130 195 C 205 195, 245 145, 335 135" fill="none" stroke="rgba(34,211,238,0.45)" strokeWidth="1.8" markerEnd="url(#arrow-cyan)" />
            <path d="M 130 195 C 205 195, 250 255, 355 255" fill="none" stroke="rgba(168,85,247,0.45)" strokeWidth="1.8" markerEnd="url(#arrow-violet)" />
            <path d="M 130 275 C 215 275, 260 260, 355 255" fill="none" stroke="rgba(34,211,238,0.5)" strokeWidth="1.8" markerEnd="url(#arrow-cyan)" />

            <path d="M 365 135 C 480 105, 680 105, 805 115" fill="none" stroke="rgba(148,163,184,0.4)" strokeWidth="1.8" markerEnd="url(#arrow-slate)" />
            <path d="M 365 135 C 455 135, 485 180, 580 185" fill="none" stroke="url(#flow-grad-risk)" strokeWidth="2.4" strokeDasharray="6 3" markerEnd="url(#arrow-risk)" filter="url(#subtle-glow)" />

            <path d="M 385 255 C 475 255, 500 195, 580 185" fill="none" stroke="url(#flow-grad-risk)" strokeWidth="2.2" strokeDasharray="6 3" markerEnd="url(#arrow-risk)" />
            <path d="M 385 255 C 465 255, 505 285, 605 285" fill="none" stroke="rgba(168,85,247,0.45)" strokeWidth="1.8" markerEnd="url(#arrow-violet)" />

            <path d="M 615 185 C 690 185, 730 215, 805 215" fill="none" stroke="#f43f5e" strokeWidth="2.6" strokeOpacity="0.9" markerEnd="url(#arrow-risk)" filter="url(#subtle-glow)" />
            <path d="M 615 185 C 685 170, 730 130, 805 115" fill="none" stroke="rgba(168,85,247,0.45)" strokeWidth="1.8" markerEnd="url(#arrow-violet)" />
            <path d="M 625 285 C 705 285, 745 295, 805 295" fill="none" stroke="rgba(148,163,184,0.4)" strokeWidth="1.8" markerEnd="url(#arrow-slate)" />

            <circle r="2.5" fill="#22d3ee" filter="url(#subtle-glow)">
              <animateMotion dur="5.2s" repeatCount="indefinite" path="M 130 115 C 195 115, 240 135, 335 135" />
            </circle>
            <circle r="2.5" fill="#f43f5e" filter="url(#subtle-glow)">
              <animateMotion dur="4.2s" repeatCount="indefinite" path="M 365 135 C 455 135, 485 180, 580 185" />
            </circle>
            <circle r="2.5" fill="#f43f5e" filter="url(#subtle-glow)">
              <animateMotion dur="3.8s" repeatCount="indefinite" path="M 615 185 C 690 185, 730 215, 805 215" />
            </circle>
            <circle r="2.2" fill="#a855f7">
              <animateMotion dur="6s" repeatCount="indefinite" path="M 130 275 C 215 275, 260 260, 355 255" />
            </circle>
          </g>

          {/* 3. Schematic Nodes */}
          <g className="flow-nodes">
            <g className="flow-node">
              <circle cx="120" cy="115" r="7" fill="#0f1c3d" stroke="#22d3ee" strokeWidth="2.2" />
              <text x="120" y="96" textAnchor="middle" fill="#f8fafc" fontSize="12" fontWeight="600" fontFamily="var(--font-mono, monospace)">
                Inflow Source A
              </text>
            </g>
            <g className="flow-node">
              <circle cx="120" cy="195" r="7" fill="#0f1c3d" stroke="#22d3ee" strokeWidth="2.2" />
              <text x="120" y="176" textAnchor="middle" fill="#f8fafc" fontSize="12" fontWeight="600" fontFamily="var(--font-mono, monospace)">
                Inflow Source B
              </text>
            </g>
            <g className="flow-node">
              <circle cx="120" cy="275" r="7" fill="#0f1c3d" stroke="#22d3ee" strokeWidth="2.2" />
              <text x="120" y="256" textAnchor="middle" fill="#f8fafc" fontSize="12" fontWeight="600" fontFamily="var(--font-mono, monospace)">
                Exchange Deposit
              </text>
            </g>

            <g className="flow-node">
              <circle cx="350" cy="135" r="8" fill="#181335" stroke="#a855f7" strokeWidth="2.2" />
              <text x="350" y="115" textAnchor="middle" fill="#f8fafc" fontSize="12" fontWeight="600" fontFamily="var(--font-mono, monospace)">
                Hop 1 Relay
              </text>
            </g>
            <g className="flow-node">
              <circle cx="370" cy="255" r="8" fill="#181335" stroke="#a855f7" strokeWidth="2.2" />
              <text x="370" y="235" textAnchor="middle" fill="#f8fafc" fontSize="12" fontWeight="600" fontFamily="var(--font-mono, monospace)">
                Consolidation Relay
              </text>
            </g>

            {/* Target Flagged Hub */}
            <g className="flow-node-target">
              <circle cx="598" cy="185" r="16" fill="none" stroke="rgba(244,63,94,0.4)" strokeWidth="1.4" strokeDasharray="4 3" />
              <circle cx="598" cy="185" r="9.5" fill="#310f1c" stroke="#f43f5e" strokeWidth="2.8" filter="url(#subtle-glow)" />
              <text x="598" y="156" textAnchor="middle" fill="#fca5a5" fontWeight="700" fontSize="13" fontFamily="var(--font-mono, monospace)">
                {targetLabel}
              </text>
              <text x="598" y="218" textAnchor="middle" fill="#fcd34d" fontWeight="600" fontSize="11" fontFamily="var(--font-mono, monospace)">
                EVIDENTIARY SINK
              </text>
            </g>

            <g className="flow-node">
              <circle cx="615" cy="285" r="7" fill="#181335" stroke="#a855f7" strokeWidth="2" />
              <text x="615" y="308" textAnchor="middle" fill="#cbd5e1" fontSize="11.5" fontFamily="var(--font-mono, monospace)">
                Hop 2 Relay
              </text>
            </g>

            <g className="flow-node">
              <circle cx="815" cy="115" r="7.5" fill="#0f1c3d" stroke="#22d3ee" strokeWidth="2.2" />
              <text x="815" y="96" textAnchor="middle" fill="#f8fafc" fontSize="12" fontWeight="600" fontFamily="var(--font-mono, monospace)">
                Licit Settlement
              </text>
            </g>
            <g className="flow-node">
              <circle cx="815" cy="215" r="8" fill="#310f1c" stroke="#f43f5e" strokeWidth="2.5" />
              <text x="815" y="196" textAnchor="middle" fill="#fca5a5" fontSize="12" fontWeight="600" fontFamily="var(--font-mono, monospace)">
                Flagged Sink
              </text>
            </g>
            <g className="flow-node">
              <circle cx="815" cy="295" r="6.5" fill="#131c31" stroke="#94a3b8" strokeWidth="2" />
              <text x="815" y="318" textAnchor="middle" fill="#94a3b8" fontSize="11.5" fontFamily="var(--font-mono, monospace)">
                Unclassified
              </text>
            </g>
          </g>
        </svg>
      </div>

      <div className="flow-legend">
        <div className="legend-items">
          <span><i style={{ background: cyan }} /> Inflow Source</span>
          <span><i style={{ background: violet }} /> Aggregation Relay</span>
          <span><i style={{ background: coral }} /> Flagged Entity Hub</span>
          <span><i style={{ background: 'rgba(148, 163, 184, 0.6)' }} /> Directed Flow Path</span>
          <span><i style={{ background: coral, borderStyle: 'dashed' }} /> Suspicious Corridor</span>
        </div>
        <div className="legend-meta">
          <span>ILLUSTRATIVE TOPOLOGY SCHEMATIC · DYNAMIC DATA RESOLVED VIA API</span>
        </div>
      </div>
    </div>
  )
}

/* =========================================================================
   Distinctive Asymmetric Hero Section (100% Dynamic Metadata, Zero Hardcoding)
   ========================================================================= */
function Hero({
  overviewData,
  topEntityId,
}: {
  overviewData?: ApiRecord
  topEntityId?: string
}) {
  const datasetOverview = (overviewData?.dataset_overview as ApiRecord) || undefined
  const topology = (overviewData?.graph_topology as ApiRecord) || undefined

  const evalSplit = datasetOverview?.test_timesteps ? `Timesteps ${datasetOverview.test_timesteps}` : '—'
  const supervisedCount = datasetOverview?.test_labeled_count
    ? `${Number(datasetOverview.test_labeled_count).toLocaleString()} Labeled`
    : '—'
  const baseIllicit = overviewData?.illicit_rate
    ? `${overviewData.illicit_rate} Base Rate`
    : datasetOverview?.test_illicit_prevalence_pct
    ? `${Number(datasetOverview.test_illicit_prevalence_pct).toFixed(2)}% Base Rate`
    : '—'
  const communities = topology?.louvain_communities
    ? `${topology.louvain_communities} Communities`
    : 'Graph Verified'
  const relCount = overviewData?.relationships ? Number(overviewData.relationships).toLocaleString() : 'observed'

  return (
    <section className="forensic-hero">
      <div className="hero-col-left">
        <div className="hero-eyebrow">
          <span className="live-dot" />
          <span>BLOCKCHAIN FORENSICS · ELLIPTIC++ BENCHMARK</span>
        </div>

        <h1 className="hero-heading">
          Trace the flow<span className="headline-period">.</span>
        </h1>

        <p className="hero-description">
          Follow continuous evidentiary signals across {relCount} directed Bitcoin transaction edges in held-out timesteps, evaluating graph neural network embeddings against supervised benchmark labels.
        </p>

        <div className="hero-meta-grid">
          <div className="meta-tile">
            <span className="meta-kicker">EVALUATION SPLIT</span>
            <strong>{evalSplit}</strong>
          </div>
          <div className="meta-tile">
            <span className="meta-kicker">SUPERVISED SUPPORT</span>
            <strong>{supervisedCount}</strong>
          </div>
          <div className="meta-tile">
            <span className="meta-kicker">BASE ILLICIT PREVALENCE</span>
            <strong className="text-coral">{baseIllicit}</strong>
          </div>
          <div className="meta-tile">
            <span className="meta-kicker">TOPOLOGY PARTITIONS</span>
            <strong className="text-cyan">{communities}</strong>
          </div>
        </div>
      </div>

      <div className="hero-col-right">
        <FlowField topEntityId={topEntityId} />
      </div>
    </section>
  )
}

/* =========================================================================
   View 01: Overview (100% Dynamic KPI Blocks + Clear Model-Insights Section)
   ========================================================================= */
function Overview({
  overviewData,
  modelsData,
  suspiciousData,
  loading,
  error,
  retry,
  onNavigate,
}: {
  overviewData?: ApiRecord
  modelsData?: ApiRecord | ApiRecord[]
  suspiciousData?: ApiRecord | ApiRecord[]
  loading: boolean
  error?: Error
  retry: () => void
  onNavigate: (view: ViewId) => void
}) {
  const row = overviewData
  const datasetOverview = (row?.dataset_overview as ApiRecord) || undefined
  const suspiciousRecords = records(suspiciousData)
  const topSuspicious = suspiciousRecords[0]
  const topEntityId = topSuspicious ? String(pick(topSuspicious, ['id', 'transaction_id', 'txId'])) : undefined

  // Derive Champion Model dynamically from /api/models
  const modelList = Array.isArray((modelsData as any)?.models)
    ? (modelsData as any).models
    : records(modelsData)
  const championModel = modelList.find((m: any) => m.id === 'risk_scorer') || modelList[0]

  const champPrAuc = championModel
    ? num(championModel, ['pr_auc', 'benchmark_pr_auc', 'auc'], 4)
    : '—'
  const champRecall = championModel?.recall !== undefined
    ? `${(Number(championModel.recall) * 100).toFixed(2)}%`
    : '—'
  const champPrecision = championModel?.precision !== undefined
    ? `${(Number(championModel.precision) * 100).toFixed(2)}%`
    : '—'
  const champAccuracy = championModel?.accuracy !== undefined
    ? `${(Number(championModel.accuracy) * 100).toFixed(2)}%`
    : '—'

  return (
    <div className="view-overview">
      <Hero overviewData={row} topEntityId={topEntityId} />

      {loading ? (
        <Loading />
      ) : error ? (
        <ErrorState error={error} retry={retry} />
      ) : (
        <>
          {/* Four Readable Dynamic KPI Blocks */}
          <div className="kpi-quartet">
            <motion.div className="kpi-card" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
              <i className="kpi-accent" style={{ background: cyan }} />
              <span className="kpi-label">Observed Transactions</span>
              <strong className="kpi-value">{formatInt(row?.total_transactions || row?.transactions)}</strong>
              <span className="kpi-hint">Full dataset transaction nodes</span>
            </motion.div>

            <motion.div className="kpi-card" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
              <i className="kpi-accent" style={{ background: violet }} />
              <span className="kpi-label">Directed Relationships</span>
              <strong className="kpi-value">{formatInt(row?.relationships || row?.edges)}</strong>
              <span className="kpi-hint">Directed transaction flow edges</span>
            </motion.div>

            <motion.div className="kpi-card" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
              <i className="kpi-accent" style={{ background: emerald }} />
              <span className="kpi-label">Held-Out Test Set</span>
              <strong className="kpi-value">{formatInt(row?.test_set || datasetOverview?.held_out_test_transactions)}</strong>
              <span className="kpi-hint">
                {datasetOverview?.test_timesteps ? `Transactions in Timesteps ${datasetOverview.test_timesteps}` : 'Held-out test partition'}
              </span>
            </motion.div>

            <motion.div className="kpi-card" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
              <i className="kpi-accent" style={{ background: coral }} />
              <span className="kpi-label">Base Illicit Rate</span>
              <strong className="kpi-value text-coral">{text(row, ['illicit_rate', 'suspicious_rate'])}</strong>
              <span className="kpi-hint">
                {datasetOverview?.test_illicit_count && datasetOverview?.test_labeled_count
                  ? `${formatInt(datasetOverview.test_illicit_count)} illicit in ${formatInt(datasetOverview.test_labeled_count)} labeled test`
                  : 'Empirical test prevalence'}
              </span>
            </motion.div>
          </div>

          {/* ONE Clear Model-Insights Section (Zero demo data, derived purely from API) */}
          <Card className="overview-summary-card">
            <SectionHead
              eyebrow="Empirical Detection Benchmark &amp; Risk Composition"
              title="Model Intelligence &amp; Detection Insights"
              meta={datasetOverview?.test_timesteps ? `TIMESTEPS ${datasetOverview.test_timesteps} BENCHMARK` : 'BENCHMARK'}
            />

            <div className="summary-split-grid">
              {/* Champion Architecture Readout (Strictly from /api/models) */}
              <div className="champion-readout">
                <div>
                  <div className="champion-header">
                    <span className="champion-title">{text(championModel, ['name', 'model', 'model_name'])}</span>
                    <span className="champion-badge">{text(championModel, ['type', 'architecture_type'], 'Evaluator')}</span>
                  </div>
                  <p className="champion-desc">
                    {text(championModel, ['description', 'architecture', 'desc'], 'Empirical model metrics evaluated across held-out evaluation timesteps.')}
                  </p>
                </div>

                <div className="champion-metrics-grid">
                  <div className="champion-metric-box highlight">
                    <span className="metric-label">PR-AUC</span>
                    <strong className="metric-val text-cyan">{champPrAuc}</strong>
                  </div>
                  <div className="champion-metric-box">
                    <span className="metric-label">Recall</span>
                    <strong className="metric-val">{champRecall}</strong>
                  </div>
                  <div className="champion-metric-box">
                    <span className="metric-label">Precision</span>
                    <strong className="metric-val">{champPrecision}</strong>
                  </div>
                  <div className="champion-metric-box">
                    <span className="metric-label">Accuracy</span>
                    <strong className="metric-val">{champAccuracy}</strong>
                  </div>
                </div>
              </div>

              {/* Top Flagged Entities (Derived strictly from /api/suspicious) */}
              <div className="summary-queue">
                <div className="queue-header">
                  <span className="queue-title">Top Flagged Test Entities</span>
                  <Label>Ranked by Calibrated Risk</Label>
                </div>

                {!suspiciousRecords.length ? (
                  <Empty message="No suspicious transaction records currently returned by /api/suspicious." />
                ) : (
                  <div className="queue-list">
                    {suspiciousRecords.slice(0, 4).map((record, index) => {
                      const txId = text(record, ['id', 'transaction_id', 'txId'])
                      const cRisk = num(record, ['calibrated_risk', 'calibratedRisk', 'calibrated_illicit_risk'])
                      const prob = num(record, ['probability', 'model_probability'])

                      return (
                        <div className="queue-row" key={txId || index}>
                          <div className="queue-entity">
                            <Target size={16} className="text-coral" />
                            <span>Tx {txId}</span>
                          </div>
                          <div className="queue-stats">
                            <span className="queue-prob font-mono">Prob {prob}</span>
                            <span className="status-pill status-high font-mono">{cRisk} RISK</span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}

                <button className="queue-btn" onClick={() => onNavigate('explorer')}>
                  <span>Inspect All {suspiciousRecords.length} Flagged Transactions in Risk Explorer</span>
                  <ArrowRight size={16} />
                </button>
              </div>
            </div>
          </Card>
        </>
      )}
    </div>
  )
}

/* =========================================================================
   View 02: Risk Explorer (100% Dynamic Accountable Table)
   ========================================================================= */
function Explorer({
  rows,
  loading,
  error,
  retry,
  query,
  setQuery,
}: {
  rows: ApiRecord[]
  loading: boolean
  error?: Error
  retry: () => void
  query: string
  setQuery: (value: string) => void
}) {
  return (
    <div className="workbench-view">
      <div className="view-masthead">
        <Label>Investigation Workbench / 02</Label>
        <h1>Risk Explorer</h1>
        <p>Dense, accountable review of empirical model predictions across the held-out test transactions.</p>
      </div>

      <Card className="workbench-card">
        <div className="workbench-toolbar">
          <div className="toolbar-stats">
            <Label>Test Period Suspicious Records</Label>
            <h2>{loading ? 'Synchronizing backend rankings...' : `${rows.length} visible test records`}</h2>
          </div>
          <div className="table-search-bar">
            <Search size={18} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search by Transaction ID or Address..."
            />
            {query && (
              <button className="clear-search-btn" onClick={() => setQuery('')}>
                Clear
              </button>
            )}
          </div>
        </div>

        {loading ? (
          <Loading />
        ) : error ? (
          <ErrorState error={error} retry={retry} />
        ) : !rows.length ? (
          <Empty message="No matching test transactions found for this query." />
        ) : (
          <div className="table-wrap">
            <table className="dense-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Transaction ID</th>
                  <th>Timestep</th>
                  <th>Model Probability</th>
                  <th>Calibrated Risk</th>
                  <th>Model Architecture</th>
                  <th>Total Degree</th>
                  <th>Risk Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((record, index) => {
                  const cRisk = numberValue(record, ['calibrated_risk', 'calibratedRisk', 'calibrated_illicit_risk']) ?? 0
                  const isHigh = cRisk >= 0.5
                  const isElevated = cRisk >= 0.2 && cRisk < 0.5
                  const statusClass = isHigh ? 'status-high' : isElevated ? 'status-elevated' : 'status-monitored'

                  return (
                    <tr key={String(pick(record, ['id', 'transaction_id', 'txId']) ?? index)}>
                      <td className="font-mono text-muted font-bold">#{text(record, ['rank']) || index + 1}</td>
                      <td className="font-mono text-cyan font-bold">{text(record, ['id', 'transaction_id', 'txId'])}</td>
                      <td className="font-mono">{text(record, ['timestep', 'Time step'])}</td>
                      <td className="font-mono">{num(record, ['probability', 'model_probability'])}</td>
                      <td className={`font-mono font-bold ${isHigh ? 'text-coral' : isElevated ? 'text-amber' : 'text-cyan'}`}>
                        {num(record, ['calibrated_risk', 'calibratedRisk', 'calibrated_illicit_risk'])}
                      </td>
                      <td>
                        <span className="model-chip">{text(record, ['model', 'model_name'])}</span>
                      </td>
                      <td className="font-mono">{text(record, ['total_degree'])}</td>
                      <td>
                        <span className={`status-pill ${statusClass}`}>{text(record, ['status', 'label'])}</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}

/* =========================================================================
   View 03: Model Lab (100% Dynamic Model Cards from /api/models)
   ========================================================================= */
function ModelLab({
  data,
  loading,
  error,
  retry,
}: {
  data?: ApiRecord | ApiRecord[]
  loading: boolean
  error?: Error
  retry: () => void
}) {
  const rows = Array.isArray((data as any)?.models) ? (data as any).models : records(data)

  return (
    <div className="studio-view">
      <div className="view-masthead">
        <Label>Evaluation Studio / 03</Label>
        <h1>Model Lab</h1>
        <p>
          Empirical held-out benchmark results on Timesteps 35–49. No fabricated confusion matrices or speculative metrics.
        </p>
      </div>

      {loading ? (
        <Loading />
      ) : error ? (
        <ErrorState error={error} retry={retry} />
      ) : !rows.length ? (
        <Empty message="Model metrics unavailable until /api/models returns data." />
      ) : (
        <div className="model-studio-grid">
          {rows.map((record: ApiRecord, index: number) => {
            const prAuc = num(record, ['pr_auc', 'benchmark_pr_auc', 'auc', 'roc_auc'])
            const prec = num(record, ['precision'])
            const rec = num(record, ['recall'])
            const f1 = num(record, ['f1', 'f1_score'])
            const isChampion = record.id === 'risk_scorer'

            return (
              <Card key={index} className={`model-tile-card ${isChampion ? 'champion-card' : ''}`}>
                <div className="tile-top">
                  <div className="tile-index">0{index + 1}</div>
                  <div className="tile-badge">{text(record, ['type', 'architecture_type'])}</div>
                </div>

                <div className="tile-header">
                  <div className="tile-title">{text(record, ['name', 'model', 'model_name'])}</div>
                  <p className="tile-desc">{text(record, ['description', 'desc', 'architecture'])}</p>
                </div>

                <div className="metric-quartet-grid">
                  <div className="metric-box">
                    <span className="metric-box-label">Precision</span>
                    <strong className="metric-box-value">{prec}</strong>
                  </div>
                  <div className="metric-box">
                    <span className="metric-box-label">Recall</span>
                    <strong className="metric-box-value">{rec}</strong>
                  </div>
                  <div className="metric-box">
                    <span className="metric-box-label">F1 Score</span>
                    <strong className="metric-box-value">{f1}</strong>
                  </div>
                  <div className="metric-box highlight-box">
                    <span className="metric-box-label">PR-AUC</span>
                    <strong className="metric-box-value text-cyan">{prAuc}</strong>
                  </div>
                </div>
              </Card>
            )
          })}
        </div>
      )}
    </div>
  )
}

/* =========================================================================
   View 04: Network Explorer (Dedicated Dynamic Ego-Network Inspector)
   ========================================================================= */
function NetworkPage({
  data,
  loading,
  error,
  retry,
}: {
  data?: ApiRecord | ApiRecord[]
  loading: boolean
  error?: Error
  retry: () => void
}) {
  const rows = records(data)
  const centerNode = (data as any)?.center_node
  const centerTxId = centerNode ? String(centerNode.id || centerNode.txId) : undefined

  return (
    <div className="immersive-view">
      <div className="view-masthead">
        <Label>Graph Investigation / 04</Label>
        <h1>Network Explorer</h1>
        <p>Interactive neighborhood and transaction transfer topology for flagged test entities.</p>
      </div>

      {loading ? (
        <Loading />
      ) : error ? (
        <ErrorState error={error} retry={retry} />
      ) : (
        <Card className="immersive-network-card">
          <SectionHead
            eyebrow="Ego-Network Neighborhood"
            title="Structural Adjacency &amp; Directional Flow"
            meta={rows.length ? `${rows.length} BACKEND NODES RESOLVED` : 'IDLE'}
          />
          <FlowField topEntityId={centerTxId} />
        </Card>
      )}
    </div>
  )
}

/* =========================================================================
   View 05: Live Inference Console (Dynamic Presets from /api/suspicious)
   ========================================================================= */
function Inference({
  score,
  suspiciousData,
  loading,
  error,
  retry,
}: {
  score?: ApiRecord | ApiRecord[]
  suspiciousData?: ApiRecord | ApiRecord[]
  loading: boolean
  error?: Error
  retry: () => void
}) {
  const row = records(score)[0]
  const [selectedModel, setSelectedModel] = useState<'gcn' | 'graphsage' | 'gru' | 'logistic'>('gcn')
  const [txInput, setTxInput] = useState('')
  const [scoring, setScoring] = useState(false)
  const [customScore, setCustomScore] = useState<ApiRecord | null>(null)

  // Derive dynamic quick presets from real suspicious data
  const suspiciousRows = records(suspiciousData)
  const dynamicPresets = useMemo(() => {
    return suspiciousRows.slice(0, 3).map((item) => String(pick(item, ['id', 'transaction_id', 'txId'])))
  }, [suspiciousRows])

  // Automatically default input to top real entity once loaded
  useEffect(() => {
    if (!txInput && dynamicPresets.length > 0) {
      setTxInput(dynamicPresets[0])
    }
  }, [dynamicPresets, txInput])

  const activeRecord = customScore || row

  const handleScore = async () => {
    if (!txInput.trim()) return
    setScoring(true)
    try {
      const res = await fetchBackend<ApiRecord>('/api/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tx_id: parseInt(txInput, 10), model: selectedModel }),
      })
      setCustomScore(res)
    } catch (e) {
      console.error(e)
    } finally {
      setScoring(false)
    }
  }

  const cRisk = numberValue(activeRecord, ['calibrated_risk', 'calibratedRisk', 'risk_score']) ?? 0
  const prob = numberValue(activeRecord, ['probability', 'model_probability']) ?? 0
  const isHigh = cRisk >= 0.5
  const isElevated = cRisk >= 0.2 && cRisk < 0.5

  return (
    <div className="inference-view">
      <div className="view-masthead">
        <Label>Scoring Console / 05</Label>
        <h1>Live Inference</h1>
        <p>On-demand evaluation across the held-out test transactions with separate model probabilities and calibrated risk.</p>
      </div>

      <div className="inference-split-grid">
        <Card className="inference-input-card">
          <SectionHead eyebrow="Inference Parameters" title="Query Specification" meta="DUAL-STACK API" />

          <div className="inference-form">
            <div className="form-group">
              <label>Model Architecture</label>
              <div className="model-selector-tabs">
                {[
                  { id: 'gcn', label: 'GCN' },
                  { id: 'graphsage', label: 'GraphSAGE' },
                  { id: 'gru', label: 'Temporal GRU' },
                  { id: 'logistic', label: 'Logistic' },
                ].map(({ id, label }) => (
                  <button
                    key={id}
                    type="button"
                    className={`model-tab-btn ${selectedModel === id ? 'active' : ''}`}
                    onClick={() => setSelectedModel(id as any)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label>Transaction ID (Held-Out Test Split)</label>
              <div className="tx-input-wrap">
                <input
                  type="text"
                  value={txInput}
                  onChange={(e) => setTxInput(e.target.value)}
                  placeholder="Enter Transaction ID..."
                />
                <button
                  type="button"
                  className="score-submit-btn"
                  onClick={handleScore}
                  disabled={scoring || !txInput.trim()}
                >
                  {scoring ? <RefreshCw size={16} className="spin" /> : <Zap size={16} />}
                  <span>{scoring ? 'Evaluating...' : 'Score Entity'}</span>
                </button>
              </div>
            </div>

            <div className="preset-quick-select">
              <span>Dynamic Quick Presets:</span>
              {!dynamicPresets.length ? (
                <span className="text-muted">Loading from API...</span>
              ) : (
                dynamicPresets.map((id, idx) => (
                  <button key={id} type="button" onClick={() => setTxInput(id)}>
                    Rank #{idx + 1} ({id})
                  </button>
                ))
              )}
            </div>
          </div>
        </Card>

        <Card className="inference-result-card">
          <SectionHead eyebrow="Verified Artifact Bridge" title="Evaluation Output" meta="/api/score" />

          {loading ? (
            <Loading />
          ) : error ? (
            <ErrorState error={error} retry={retry} />
          ) : !activeRecord ? (
            <Empty message="Inference unavailable until /api/score returns a response." />
          ) : (
            <div className="score-readout-wrap">
              <div className="score-primary-gauges">
                <div className="gauge-box">
                  <span className="gauge-label">Raw Model Probability</span>
                  <strong className="gauge-value text-cyan">{prob.toFixed(4)}</strong>
                  <span className="gauge-sub">Continuous classification signal</span>
                </div>

                <div className="gauge-box">
                  <span className="gauge-label">Calibrated Illicit Risk</span>
                  <strong className={`gauge-value ${isHigh ? 'text-coral' : isElevated ? 'text-amber' : 'text-cyan'}`}>
                    {cRisk.toFixed(4)}
                  </strong>
                  <span className="gauge-sub">Validation-calibrated risk</span>
                </div>
              </div>

              <div className="inference-summary-strip">
                <div className="strip-item">
                  <span>Evaluated Entity</span>
                  <strong className="font-mono text-cyan">Tx {text(activeRecord, ['id', 'transaction_id', 'txId'])}</strong>
                </div>
                <div className="strip-item">
                  <span>Model Selector</span>
                  <strong>{text(activeRecord, ['model', 'requested_model'])}</strong>
                </div>
                <div className="strip-item">
                  <span>Risk Classification</span>
                  <span className={`status-pill ${isHigh ? 'status-high' : isElevated ? 'status-elevated' : 'status-monitored'}`}>
                    {isHigh ? 'HIGH RISK' : isElevated ? 'ELEVATED RISK' : 'MONITORED'}
                  </span>
                </div>
              </div>

              <div className="notice-box">
                <ShieldCheck size={18} className="text-cyan" />
                <p>Model outputs are empirical evidentiary signals, not definitive proof of financial crime.</p>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}

/* =========================================================================
   Primary Root Shell & Dock Navigation
   ========================================================================= */
export default function IntelligenceDashboard() {
  const [page, setPage] = useState<ViewId>('overview')
  const [query, setQuery] = useState('')

  const overview = useSWR('/api/overview', (path) => fetchBackend<ApiRecord>(path))
  const suspicious = useSWR('/api/suspicious', (path) => fetchBackend<ApiRecord | ApiRecord[]>(path))
  const models = useSWR('/api/models', (path) => fetchBackend<ApiRecord | ApiRecord[]>(path))
  const network = useSWR('/api/network', (path) => fetchBackend<ApiRecord | ApiRecord[]>(path))
  const score = useSWR('/api/score', (path) => fetchBackend<ApiRecord | ApiRecord[]>(path))

  const isConnected = Boolean(overview.data || models.data || suspicious.data)

  const filteredRows = useMemo(
    () =>
      records(suspicious.data).filter((record) =>
        text(record, ['id', 'transaction_id', 'address', 'hash', 'txId'])
          .toLowerCase()
          .includes(query.toLowerCase())
      ),
    [suspicious.data, query]
  )

  const retry = () => [overview, suspicious, models, network, score].forEach((feed: Feed) => feed.mutate())

  return (
    <div className="command-shell">
      {/* Top Header Bar */}
      <header className="brand-strip">
        <div className="brand-lockup">
          <div className="brand-mark">
            <span />
            <span />
            <span />
          </div>
          <div>
            <strong>
              elliptic<span>++</span>
            </strong>
            <small>AI FINANCIAL INTELLIGENCE</small>
          </div>
        </div>

        <div className="top-command-controls">
          <div className="brand-meta">
            <span className="live-dot" style={{ background: isConnected ? cyan : amber }} />
            <span>{isConnected ? 'BACKEND CONNECTED · DUAL-STACK API' : 'CONNECTING TO BACKEND...'}</span>
          </div>

          <button className="sync-btn" onClick={retry}>
            <RefreshCw size={14} />
            <span>Sync feeds</span>
          </button>
        </div>
      </header>

      {/* Main Responsive Content Surface (Uses 96-98% viewport width) */}
      <main className="command-main">
        <AnimatePresence mode="wait">
          <motion.div
            key={page}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
          >
            {page === 'overview' && (
              <Overview
                overviewData={overview.data}
                modelsData={models.data}
                suspiciousData={suspicious.data}
                loading={overview.isLoading || models.isLoading}
                error={overview.error}
                retry={retry}
                onNavigate={setPage}
              />
            )}
            {page === 'explorer' && (
              <Explorer
                rows={filteredRows}
                loading={suspicious.isLoading}
                error={suspicious.error}
                retry={retry}
                query={query}
                setQuery={setQuery}
              />
            )}
            {page === 'models' && (
              <ModelLab data={models.data} loading={models.isLoading} error={models.error} retry={retry} />
            )}
            {page === 'network' && (
              <NetworkPage data={network.data} loading={network.isLoading} error={network.error} retry={retry} />
            )}
            {page === 'inference' && (
              <Inference
                score={score.data}
                suspiciousData={suspicious.data}
                loading={score.isLoading}
                error={score.error}
                retry={retry}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Floating Bottom Navigation Dock (Aligned, Larger, Legible, Compact, Never Overlaps) */}
      <nav className="floating-dock" aria-label="Primary navigation">
        {nav.map(({ id, label, icon: Icon, index }) => (
          <button
            key={id}
            className={`dock-btn ${page === id ? 'active' : ''}`}
            onClick={() => setPage(id)}
            aria-label={label}
            aria-current={page === id ? 'page' : undefined}
          >
            <span className="dock-index">{index}</span>
            <Icon size={18} />
            <span className="dock-label">{label}</span>
          </button>
        ))}
      </nav>
    </div>
  )
}
