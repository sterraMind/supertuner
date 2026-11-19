import { useId, type ReactNode } from "react";

/**
 * Form primitives. Every control renders a real <label for>, so clicking the
 * label focuses the input and screen readers announce the field name.
 */

export interface FieldProps {
  label: string;
  hint?: string;
  error?: string;
  htmlFor?: string;
  children: ReactNode;
  className?: string;
}

export function Field({ label, hint, error, htmlFor, children, className }: FieldProps) {
  return (
    <div className={className ? `field ${className}` : "field"}>
      <label className="field-label" htmlFor={htmlFor}>
        {label}
      </label>
      {children}
      {hint && !error ? <p className="field-hint">{hint}</p> : null}
      {error ? (
        <p className="field-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}

export interface NumberFieldProps {
  label: string;
  value: number;
  onChange(value: number): void;
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
  disabled?: boolean;
  /** Unit shown after the label, e.g. "MiB". */
  unit?: string;
}

export function NumberField({ label, value, onChange, min, max, step, hint, disabled, unit }: NumberFieldProps) {
  const id = useId();
  return (
    <Field label={unit ? `${label} (${unit})` : label} hint={hint} htmlFor={id}>
      <input
        id={id}
        className="input num"
        type="number"
        inputMode="numeric"
        value={Number.isFinite(value) ? value : 0}
        min={min}
        max={max}
        step={step ?? 1}
        disabled={disabled}
        onChange={(event) => {
          const next = event.target.valueAsNumber;
          onChange(Number.isFinite(next) ? next : 0);
        }}
      />
    </Field>
  );
}

export interface TextFieldProps {
  label: string;
  value: string;
  onChange(value: string): void;
  placeholder?: string;
  hint?: string;
  disabled?: boolean;
  mono?: boolean;
  type?: "text" | "search" | "url";
}

export function TextField({
  label,
  value,
  onChange,
  placeholder,
  hint,
  disabled,
  mono,
  type = "text",
}: TextFieldProps) {
  const id = useId();
  return (
    <Field label={label} hint={hint} htmlFor={id}>
      <input
        id={id}
        className={mono ? "input mono" : "input"}
        type={type}
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
    </Field>
  );
}

export interface TextAreaFieldProps {
  label: string;
  value: string;
  onChange(value: string): void;
  placeholder?: string;
  hint?: string;
  rows?: number;
}

export function TextAreaField({ label, value, onChange, placeholder, hint, rows = 3 }: TextAreaFieldProps) {
  const id = useId();
  return (
    <Field label={label} hint={hint} htmlFor={id}>
      <textarea
        id={id}
        className="textarea"
        rows={rows}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </Field>
  );
}

export interface SelectOption<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
}

export interface SelectFieldProps<T extends string> {
  label: string;
  value: T;
  options: readonly SelectOption<T>[];
  onChange(value: T): void;
  hint?: string;
  disabled?: boolean;
  placeholder?: string;
}

export function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
  hint,
  disabled,
  placeholder,
}: SelectFieldProps<T>) {
  const id = useId();
  return (
    <Field label={label} hint={hint} htmlFor={id}>
      <select
        id={id}
        className="select"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value as T)}
      >
        {placeholder ? <option value="">{placeholder}</option> : null}
        {options.map((option) => (
          <option key={option.value} value={option.value} disabled={option.disabled}>
            {option.label}
          </option>
        ))}
      </select>
    </Field>
  );
}

export interface CheckboxFieldProps {
  label: string;
  checked: boolean;
  onChange(checked: boolean): void;
  hint?: string;
  disabled?: boolean;
}

export function CheckboxField({ label, checked, onChange, hint, disabled }: CheckboxFieldProps) {
  const id = useId();
  return (
    <div className="field">
      <label className="check-row" htmlFor={id}>
        <input
          id={id}
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>
          {label}
          {hint ? <span className="field-hint"> {hint}</span> : null}
        </span>
      </label>
    </div>
  );
}

export interface SwitchFieldProps {
  label: string;
  checked: boolean;
  onChange(checked: boolean): void;
  hint?: string;
  disabled?: boolean;
}

/** Labelled switch. Uses role="switch" so the on/off state is announced. */
export function SwitchField({ label, checked, onChange, hint, disabled }: SwitchFieldProps) {
  const id = useId();
  return (
    <div className="switch-row">
      <div className="stack-sm">
        <label className="field-label" htmlFor={id}>
          {label}
        </label>
        {hint ? <p className="field-hint">{hint}</p> : null}
      </div>
      <button
        id={id}
        type="button"
        className="switch"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        disabled={disabled}
        onClick={() => onChange(!checked)}
      >
        <span className="switch-track" aria-hidden="true" data-on={checked ? "true" : "false"} />
      </button>
    </div>
  );
}

export interface CheckGroupProps {
  label: string;
  options: readonly SelectOption<string>[];
  selected: readonly string[];
  onChange(selected: string[]): void;
  hint?: string;
}

/** Multi-select as checkboxes, used for LoRA targets and export formats. */
export function CheckGroup({ label, options, selected, onChange, hint }: CheckGroupProps) {
  return (
    <fieldset className="field">
      <legend className="field-label">{label}</legend>
      <div className="check-group">
        {options.map((option) => {
          const checked = selected.includes(option.value);
          const id = `${label}-${option.value}`.replace(/\s+/g, "-").toLowerCase();
          return (
            <label className="check-row" key={option.value} htmlFor={id}>
              <input
                id={id}
                type="checkbox"
                checked={checked}
                onChange={() =>
                  onChange(checked ? selected.filter((item) => item !== option.value) : [...selected, option.value])
                }
              />
              <span>{option.label}</span>
            </label>
          );
        })}
      </div>
      {hint ? <p className="field-hint">{hint}</p> : null}
    </fieldset>
  );
}