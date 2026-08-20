import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

const defaults = {
  fill: "none",
  stroke: "currentColor",
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  strokeWidth: 1.8,
  viewBox: "0 0 24 24",
  "aria-hidden": true,
};

export function ArrowRightIcon(props: IconProps) {
  return (
    <svg {...defaults} {...props}>
      <path d="M5 12h14M13 6l6 6-6 6" />
    </svg>
  );
}

export function CheckIcon(props: IconProps) {
  return (
    <svg {...defaults} {...props}>
      <path d="m5 12 4 4L19 6" />
    </svg>
  );
}

export function ChevronIcon(props: IconProps) {
  return (
    <svg {...defaults} {...props}>
      <path d="m8 10 4 4 4-4" />
    </svg>
  );
}

export function DatabaseIcon(props: IconProps) {
  return (
    <svg {...defaults} {...props}>
      <ellipse cx="12" cy="5" rx="7" ry="3" />
      <path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" />
    </svg>
  );
}

export function DownloadIcon(props: IconProps) {
  return (
    <svg {...defaults} {...props}>
      <path d="M12 3v12m0 0 4-4m-4 4-4-4M5 21h14" />
    </svg>
  );
}

export function ExternalLinkIcon(props: IconProps) {
  return (
    <svg {...defaults} {...props}>
      <path d="M14 5h5v5M19 5l-8 8M19 13v6H5V5h6" />
    </svg>
  );
}

export function InfoIcon(props: IconProps) {
  return (
    <svg {...defaults} {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5M12 8h.01" />
    </svg>
  );
}

export function LockIcon(props: IconProps) {
  return (
    <svg {...defaults} {...props}>
      <rect x="5" y="10" width="14" height="10" rx="2" />
      <path d="M8 10V7a4 4 0 0 1 8 0v3" />
    </svg>
  );
}

export function RefreshIcon(props: IconProps) {
  return (
    <svg {...defaults} {...props}>
      <path d="M20 7v5h-5M4 17v-5h5" />
      <path d="M18.2 9A7 7 0 0 0 6.6 6.6L4 9m16 6-2.6 2.4A7 7 0 0 1 5.8 15" />
    </svg>
  );
}

export function ShieldIcon(props: IconProps) {
  return (
    <svg {...defaults} {...props}>
      <path d="M12 3 5 6v5c0 4.7 2.7 8.1 7 10 4.3-1.9 7-5.3 7-10V6l-7-3Z" />
      <path d="m9 12 2 2 4-5" />
    </svg>
  );
}

export function SparkIcon(props: IconProps) {
  return (
    <svg {...defaults} {...props}>
      <path d="m12 3 1.4 4.2a5.4 5.4 0 0 0 3.4 3.4L21 12l-4.2 1.4a5.4 5.4 0 0 0-3.4 3.4L12 21l-1.4-4.2a5.4 5.4 0 0 0-3.4-3.4L3 12l4.2-1.4a5.4 5.4 0 0 0 3.4-3.4L12 3Z" />
    </svg>
  );
}

export function UsersIcon(props: IconProps) {
  return (
    <svg {...defaults} {...props}>
      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM22 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8" />
    </svg>
  );
}
