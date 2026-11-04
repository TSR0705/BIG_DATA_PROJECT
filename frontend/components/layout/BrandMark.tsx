import Link from "next/link";

export function BrandMark() {
  return (
    <Link href="/" className="flex items-center gap-2.5 px-2 py-3 select-none">
      <div className="w-8 h-8 flex items-center justify-center shrink-0 text-[#158444]">
        <svg className="w-7 h-7" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"
            fill="#158444"
          />
          <path
            d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"
            stroke="#ffffff"
            strokeWidth="1.4"
            strokeLinecap="round"
          />
        </svg>
      </div>
      <div className="leading-tight">
        <h1 className="font-extrabold text-[17px] text-ink tracking-tight">CreditLens</h1>
        <p className="text-[11px] font-medium text-[#89938D]">Loan Default Prediction</p>
      </div>
    </Link>
  );
}
