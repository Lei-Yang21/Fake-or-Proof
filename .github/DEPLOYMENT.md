# Automatic Streamlit deployment

The workflows deploy the existing Streamlit service independently of the Vertex
AI training pipeline. Deployment settings live in GitHub Actions settings rather
than in the workflow source or this guide.

## Release behavior

A push to `main` affecting app code, its Dockerfile, app dependencies, app tests,
or the two workflows runs tests, builds an image tagged with the commit SHA, and
deploys only if checks pass. The existing Cloud Run service receives 100% of
traffic on the new revision and keeps its URL and public-access IAM policy.
The container preflights artifact integrity and real inference before opening HTTP.
The workflow then checks `/_stcore/health` with a short-lived identity token.

Changes to shared code, the bundled model/manifest, Streamlit configuration and
`pyproject.toml`/`uv.lock` trigger an application release. Changes confined to
training components, EDA, documentation or legacy templates do not. Project CI
checks the application and pipeline independently of those release filters. No
workflow starts training.
Matching pull requests run app checks without cloud credentials. Manual
**Actions > Deploy Streamlit > Run workflow** deployments must use `main`.
Refresh an existing browser session after deployment to use the updated app.

`workflow_call: {}` and `workflow_dispatch: {}` enable events with no additional
inputs; they do not represent missing configuration.

## GitHub configuration

Open **Settings > Secrets and variables > Actions** in your GitHub repository.
Create these five entries under **Variables > New repository variable**:

| Variable | Value to enter |
| --- | --- |
| `GCP_PROJECT_ID` | Your project ID, not its numeric project number |
| `GCP_REGION` | Region containing the Cloud Run service and Docker repository |
| `GCP_ARTIFACT_REPOSITORY` | Existing Artifact Registry Docker repository name |
| `GCP_CLOUD_RUN_SERVICE` | Existing Streamlit Cloud Run service name |
| `GCP_RUNTIME_SERVICE_ACCOUNT` | Full email of the app runtime service account |

Under **Secrets > New repository secret**, create these two entries:

| Secret | Value to enter |
| --- | --- |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Full resource path: `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_ID/providers/PROVIDER_ID` |
| `GCP_DEPLOY_SERVICE_ACCOUNT` | Full email of the deployment service account |

These two authentication identifiers are stored as secrets to keep their values
out of source and mask them in logs. They identify resources; they are not
private keys. Access still depends on the provider's repository/branch condition
and IAM permissions. No JSON key is needed.

If these two entries were previously created under **Variables**, recreate them
under **Secrets** with the same names and values. The workflow now uses
`secrets.GCP_WORKLOAD_IDENTITY_PROVIDER` and `secrets.GCP_DEPLOY_SERVICE_ACCOUNT`.
Remove the old variables after creating the secrets to avoid duplicate settings.

All seven settings are required. Before building or authenticating, the workflow
reports each missing setting by name without printing its value. The image path
and registry hostname are derived from the variables. There are no defaults
that silently select a different project or service.

## One-time Google Cloud setup

If federation and permissions are already configured, use their existing IDs
and skip resource creation. The default app uses its bundled verified model and no Firestore, so its runtime
identity needs neither model-bucket nor database permissions. Review and revoke
obsolete grants administratively before opening the service. Optional GCS mode
needs read-only access to the selected model object and explicit trusted URI/hash
configuration. The deployment identity needs image upload, service-update,
permission to attach that runtime identity, and service invocation for the
authenticated health check.

These commands are for Windows Command Prompt (`cmd`). Replace every `YOUR_...`
placeholder before running them. Local CMD variables are used only for setup;
GitHub does not read them.

```cmd
set "GCP_PROJECT_ID=YOUR_PROJECT_ID"
set "GCP_PROJECT_NUMBER=YOUR_NUMERIC_PROJECT_NUMBER"
set "GCP_REGION=YOUR_REGION"
set "GCP_ARTIFACT_REPOSITORY=YOUR_EXISTING_DOCKER_REPOSITORY"
set "GCP_CLOUD_RUN_SERVICE=YOUR_EXISTING_CLOUD_RUN_SERVICE"
set "GCP_RUNTIME_SERVICE_ACCOUNT=YOUR_APP_SERVICE_ACCOUNT_EMAIL"
set "GITHUB_REPOSITORY=YOUR_GITHUB_OWNER/YOUR_REPOSITORY"
set "DEPLOY_SA_NAME=github-deploy"
set "POOL_ID=github"
set "PROVIDER_ID=github"
set "GCP_DEPLOY_SERVICE_ACCOUNT=%DEPLOY_SA_NAME%@%GCP_PROJECT_ID%.iam.gserviceaccount.com"
set "GCP_WORKLOAD_IDENTITY_PROVIDER=projects/%GCP_PROJECT_NUMBER%/locations/global/workloadIdentityPools/%POOL_ID%/providers/%PROVIDER_ID%"
```

Find the numeric project number in Google Cloud project settings or with:

```cmd
gcloud projects describe "%GCP_PROJECT_ID%" --format="value(projectNumber)"
```

Run the remaining commands one at a time using an account allowed to administer
the project. Skip creation commands for resources that already exist.

Enable federation APIs and create a deployment identity:

```cmd
gcloud services enable iamcredentials.googleapis.com sts.googleapis.com --project="%GCP_PROJECT_ID%"
gcloud iam service-accounts create "%DEPLOY_SA_NAME%" --display-name="GitHub Actions deployment" --project="%GCP_PROJECT_ID%"
```

Grant access on the existing Docker repository and Cloud Run service, including
permission to invoke the health endpoint, then allow the deployment account to
attach the app runtime account:

```cmd
gcloud artifacts repositories add-iam-policy-binding "%GCP_ARTIFACT_REPOSITORY%" --location="%GCP_REGION%" --project="%GCP_PROJECT_ID%" --member="serviceAccount:%GCP_DEPLOY_SERVICE_ACCOUNT%" --role="roles/artifactregistry.writer"
gcloud run services add-iam-policy-binding "%GCP_CLOUD_RUN_SERVICE%" --region="%GCP_REGION%" --project="%GCP_PROJECT_ID%" --member="serviceAccount:%GCP_DEPLOY_SERVICE_ACCOUNT%" --role="roles/run.developer"

gcloud run services add-iam-policy-binding "%GCP_CLOUD_RUN_SERVICE%" --region="%GCP_REGION%" --project="%GCP_PROJECT_ID%" --member="serviceAccount:%GCP_DEPLOY_SERVICE_ACCOUNT%" --role="roles/run.invoker"
gcloud iam service-accounts add-iam-policy-binding "%GCP_RUNTIME_SERVICE_ACCOUNT%" --project="%GCP_PROJECT_ID%" --member="serviceAccount:%GCP_DEPLOY_SERVICE_ACCOUNT%" --role="roles/iam.serviceAccountUser"
```

Create federation restricted to your repository's `main` branch:

```cmd
gcloud iam workload-identity-pools create "%POOL_ID%" --location=global --display-name="GitHub Actions" --project="%GCP_PROJECT_ID%"
gcloud iam workload-identity-pools providers create-oidc "%PROVIDER_ID%" --location=global --workload-identity-pool="%POOL_ID%" --issuer-uri="https://token.actions.githubusercontent.com" --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" --attribute-condition="assertion.repository=='%GITHUB_REPOSITORY%' && assertion.ref=='refs/heads/main'" --project="%GCP_PROJECT_ID%"
gcloud iam service-accounts add-iam-policy-binding "%GCP_DEPLOY_SERVICE_ACCOUNT%" --project="%GCP_PROJECT_ID%" --role="roles/iam.workloadIdentityUser" --member="principalSet://iam.googleapis.com/projects/%GCP_PROJECT_NUMBER%/locations/global/workloadIdentityPools/%POOL_ID%/attribute.repository/%GITHUB_REPOSITORY%"
```

The provider's branch condition also applies to the repository-level IAM binding.
Allow time for IAM changes to propagate. Copy the five variable values and two
secret values above to the matching GitHub settings tabs.

References: [Google's federation setup](https://cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines),
[GitHub authentication action](https://github.com/google-github-actions/auth),
[GitHub Actions secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets),
[Cloud Run deployment permissions](https://cloud.google.com/run/docs/deploying),
and [authenticated Cloud Run requests](https://cloud.google.com/run/docs/authenticating/service-to-service).

## Commit and verify

Review the complete diff because project changes may already be pending. Include
`fake_or_proof/`, `.streamlit/`, the migrated model and manifest, `pyproject.toml`,
`uv.lock`, workflows, CI assets and documentation. Never add
credentials, virtualenvs or pipeline build outputs.

The default release now uses a bundled restricted checkpoint and private session
history. It does not need GCS or Firestore. No workflow rewrites an existing model bucket or revokes old IAM grants. Optional
GCS serving uses repository variables `FOP_MODEL_URI` and `FOP_MODEL_SHA256`, which
the deployment workflow validates and transfers to Cloud Run. The URI defaults to
`bundled`, whose hash comes from the release manifest. A GCS object requires the
trusted 64-character hash and read-only access for the runtime account. Existing
full-model pickle objects are incompatible; publish a known restricted checkpoint
or use the bundled release artifact.

After pushing to `main`, inspect **Actions > Deploy Streamlit**. If configuration
was incomplete, finish setup and use **Run workflow** on `main` to retry. A matching
push can deploy automatically. The workflow uses one instance, four concurrent
requests and a five-minute request timeout; the application itself serializes
model operations. Set broader ingress/admission controls according to actual
traffic before scaling an anonymous service.

## Automated release checks

The CI workflow runs on Linux and installs application/pipeline/test groups from
`uv.lock` in locked mode. Tests and utilities are kept under `.github/tests/` and
`.github/scripts/`; none are included in the runtime image. CI validates real
inference/GradCAM, session isolation, the safe artifact loader, image boundaries,
GCS preconditions and the pipeline graph/promotion logic with simulated storage.

Both runtime images are built, checked for an unprivileged user and scanned for
fixable high/critical vulnerabilities. Package versions in the universal lock are
also audited against public advisories. There is no local bootstrap benchmark,
local launch script or local demonstration dataset in the deployment path.

The server entry point verifies its artifact and runs inference before opening
HTTP on `0.0.0.0` and Cloud Run's `PORT` (8080). After deployment, the workflow
generates an identity token with the service URL as its audience and includes it
in the health request's Authorization header. The token is generated just before
the check, is masked by the authentication action, and is not written to a new
credentials file. The existing federation binding grants token generation; no
service-account key or additional GitHub secret is required. The deployment
account must have `roles/run.invoker` on a private service, as shown above.

This check preserves the service's existing access and ingress policies. A
service restricted to internal ingress still needs a runner with the necessary
network access. After deployment, inspect the interface using two independent
browser sessions. No CI check accesses production Firestore or starts training.
